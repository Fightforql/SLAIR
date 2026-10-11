import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision.transforms as T
from peft import LoraConfig, get_peft_model
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from vavae.ldm.modules.diffusionmodules.model import Encoder, Decoder
from vavae.ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from vavae.ldm.util import instantiate_from_config

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
                 vf_ckpt_path=None,
                 reverse_proj=True,
                 proj_fix=False,
                 use_lora=False,
                 lora_rank=16,
                 lora_alpha=32,
                 lora_dropout=0.0
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

        self.num_classes = 11
        #cls
        self.classifier_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier_head = torch.nn.Linear(embed_dim, self.num_classes)
        ''' 
        self.augmenter = T.Compose([
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ])
        '''
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        
        # Foundation Model
        if use_vf is not None:
            self.use_vf = use_vf
            from vavae.ldm.models.foundation_models import aux_foundation_model
            print(f"Using {use_vf} as auxiliary feature.")
            self.foundation_model = aux_foundation_model(use_vf, vf_ckpt_path)
            vf_feature_dim = self.foundation_model.feature_dim
            self.linear_proj = torch.nn.Conv2d(vf_feature_dim, embed_dim, kernel_size=1, bias=True)
            if reverse_proj:
                self.linear_proj = torch.nn.Conv2d(embed_dim, vf_feature_dim, kernel_size=1, bias=False)
        else:
            self.use_vf = None
        self.reverse_proj = reverse_proj
        self.proj_fix = proj_fix
        self.automatic_optimization = False
        # Metrics
        data_range = 2.0 
        self.val_psnr = PeakSignalNoiseRatio(data_range=data_range)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=data_range)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        #lora
        self.use_lora = use_lora
        if self.use_lora:
            print(f"Injecting PEFT LoRA (rank={lora_rank})...")
            target_modules = ["conv1", "conv2", "conv_in", "nin_shortcut","q","k","v","quant_conv", "post_quant_conv"]
            
            peft_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                modules_to_save=[] 
            )
            
            self.encoder = get_peft_model(self.encoder, peft_config)
            self.decoder = get_peft_model(self.decoder, peft_config)
            
            self.encoder.print_trainable_parameters()
            self.decoder.print_trainable_parameters()
            
        try:
            last_layer = self.decoder.base_model.model.conv_out
        except AttributeError:
            last_layer = self.decoder.conv_out
            
        print(f"Unfreezing last layer: {last_layer}")
        for param in last_layer.parameters():
            param.requires_grad = True
        self.classifier_head.requires_grad_(True)
        if self.use_vf is not None and not self.proj_fix:
            self.linear_proj.requires_grad_(True)
        
        for param in self.quant_conv.parameters():
            param.requires_grad = True
        for param in self.post_quant_conv.parameters():
            param.requires_grad = True
            

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"missing：{missing}, unexpected: {unexpected}")
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, hqs, lqs, sample_posterior=True):
        # 1. Encode HQ
        hq_posterior = self.encode(hqs)
        if sample_posterior:
            hq_z = hq_posterior.sample()
        else:
            hq_z = hq_posterior.mode()
        hq_dec = self.decode(hq_z)
        
        # 2. Encode LQ
        lq_posterior = self.encode(lqs)
        if sample_posterior:
            lq_z = lq_posterior.sample()
        else:
            lq_z = lq_posterior.mode()
        lq_dec = self.decode(lq_z)
        '''
        #  Augmentation for Contrastive Learning (Training only)
        lq_aug_feat = None
        if self.training:
            # [-1, 1] -> [0, 1] for jitter -> [-1, 1]
            lqs_01 = (lqs + 1.0) / 2.0
            lqs_aug_01 = self.augmenter(lqs_01)
            lqs_aug = (lqs_aug_01 * 2.0) - 1.0
            
            lq_aug_posterior = self.encode(lqs_aug)
            if sample_posterior:
                lq_aug_z = lq_aug_posterior.sample()
            else:
                lq_aug_z = lq_aug_posterior.mode()
            
            lq_aug_feat = self.classifier_pool(lq_aug_z).squeeze(-1).squeeze(-1)
            lq_aug_feat = F.normalize(lq_aug_feat, dim=1)

        # Feature Pooling
        lq_feat = self.classifier_pool(lq_z).squeeze(-1).squeeze(-1)
        hq_feat = self.classifier_pool(hq_z).squeeze(-1).squeeze(-1)
        
        lq_feat_norm = F.normalize(lq_feat, dim=1)
        hq_feat_norm = F.normalize(hq_feat, dim=1)
        '''
        lq_feat = self.classifier_pool(lq_z).squeeze(-1).squeeze(-1)
        hq_feat = self.classifier_pool(hq_z).squeeze(-1).squeeze(-1)
        # Classification Logits
        lq_logits = self.classifier_head(lq_feat)
        hq_logits = self.classifier_head(hq_feat)

        return hq_dec, lq_dec, hq_posterior, lq_posterior, \
               hq_logits, lq_logits

    def get_input(self, batch):
        x = batch["hq"]
        y = batch["lq"]
        label = batch["label"]
        if len(x.shape) == 3: x = x[..., None]
        if len(y.shape) == 3: y = y[..., None]
        return x, y, label

    def training_step(self, batch, batch_idx):
        hqs, lqs, labels = self.get_input(batch)
        
        hq_rec, lq_rec, hq_post, lq_post, \
        hq_logits, lq_logits = self(hqs, lqs)
        ae_opt, disc_opt = self.optimizers()
        # Generator Step
        aeloss, log_dict_ae = self.loss(
            inputs_hq=hqs, inputs_lq=lqs,
            rec_hq=hq_rec, rec_lq=lq_rec,
            post_hq=hq_post, post_lq=lq_post,
            labels=labels,
            logits_hq=hq_logits, logits_lq=lq_logits,
            optimizer_idx=0, global_step=self.global_step,
            last_layer=self.get_last_layer(), split="train"
        )
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        ae_opt.zero_grad()
        self.manual_backward(aeloss)
        ae_opt.step()

        
        discloss, log_dict_disc = self.loss(
            inputs_hq=hqs, inputs_lq=lqs,
            rec_hq=hq_rec, rec_lq=lq_rec,
            post_hq=hq_post, post_lq=lq_post,
            labels=labels,
            logits_hq=hq_logits, logits_lq=lq_logits,
            optimizer_idx=1, global_step=self.global_step,
            last_layer=self.get_last_layer(), split="train"
        )
        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        disc_opt.zero_grad()
        self.manual_backward(discloss)
        disc_opt.step()
            
    def validation_step(self, batch, batch_idx):
        hqs, lqs, labels = self.get_input(batch)
        
        hq_rec, lq_rec, hq_post, lq_post, \
        hq_logits, lq_logits = self(hqs, lqs) 
        
        aeloss, log_dict_ae = self.loss(
            inputs_hq=hqs, inputs_lq=lqs,
            rec_hq=hq_rec, rec_lq=lq_rec,
            post_hq=hq_post, post_lq=lq_post,
            labels=labels,
            logits_hq=hq_logits, logits_lq=lq_logits,
            optimizer_idx=0, global_step=self.global_step,
            last_layer=self.get_last_layer(), split="val"
        )
        
        discloss, log_dict_disc = self.loss(
            inputs_hq=hqs, inputs_lq=lqs,
            rec_hq=hq_rec, rec_lq=lq_rec,
            post_hq=hq_post, post_lq=lq_post,
            labels=labels,
            logits_hq=hq_logits, logits_lq=lq_logits,
            optimizer_idx=1, global_step=self.global_step,
            last_layer=self.get_last_layer(), split="val"
        )
        
        self.log("val/psnr", self.val_psnr(hq_rec, hqs), prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        self.log("val/ssim", self.val_ssim(hq_rec, hqs), prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        self.log("val/psnr_lq", self.val_psnr(lq_rec, lqs), prog_bar=False, logger=True, on_epoch=True, sync_dist=True)

        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        
        params_to_train = filter(lambda p: p.requires_grad, self.parameters())
        
        opt_ae = torch.optim.Adam(params_to_train, lr=lr, betas=(0.5, 0.9))
        
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        try:
            return self.decoder.conv_out.weight
        except:
            return self.decoder.base_model.model.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        hqs, lqs, labels = self.get_input(batch)
        hqs, lqs = hqs.to(self.device), lqs.to(self.device)
        
        if not only_inputs:
            hq_rec, lq_rec, hq_post, _, _, _ = self(hqs, lqs)
            if hqs.shape[1] > 3:
                hqs = self.to_rgb(hqs)
                hq_rec = self.to_rgb(hq_rec)
                lqs = self.to_rgb(lqs)
                lq_rec = self.to_rgb(lq_rec)
            log["hq_reconstructions"] = hq_rec
            log["lq_reconstructions"] = lq_rec
            log["samples"] = self.decode(torch.randn_like(hq_post.sample()))
        
        log["inputs_hq"] = hqs
        log["inputs_lq"] = lqs
        return log
    
    def to_rgb(self, x):
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x