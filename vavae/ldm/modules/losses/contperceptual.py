import torch
import torch.nn as nn
import torch.nn.functional as F
from taming.modules.losses.vqperceptual import LPIPS, NLayerDiscriminator, weights_init, hinge_d_loss, vanilla_d_loss, adopt_weight

class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", pp_style=True, vf_weight=1e2, adaptive_vf=False,
                 cls_weight=1.0, 
                 **kwargs # Absorb unused arguments from old configs
                 ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.pp_style = pp_style
        self.vf_weight = vf_weight
        self.adaptive_vf = adaptive_vf
        
        self.cls_weight = cls_weight
        #self.contrastive_weight = contrastive_weight
        #self.contrastive_temp = contrastive_temp
        self.classification_loss = nn.CrossEntropyLoss()

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    def forward(self, 
                inputs_hq, inputs_lq,     # Both inputs
                rec_hq, rec_lq,           # Both reconstructions
                post_hq, post_lq,         # Both posteriors
                labels, 
                #feat_hq, feat_lq, feat_lq_aug, # Features
                logits_hq, logits_lq,
                optimizer_idx, 
                global_step, 
                last_layer=None, cond=None, split="train",
                weights=None, enc_last_layer=None,
                **kwargs):
        
        device = inputs_hq.device
        
        inputs_all = torch.cat([inputs_hq.contiguous(), inputs_lq.contiguous()], dim=0)
        rec_all = torch.cat([rec_hq.contiguous(), rec_lq.contiguous()], dim=0)
        
        rec_loss = torch.abs(inputs_all - rec_all)
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs_all, rec_all)
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        if not self.pp_style:
            nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
            if weights is not None:
                nll_loss = nll_loss * weights
            nll_loss = nll_loss.mean()
            
            with torch.no_grad():
                bs = inputs_hq.shape[0]
                rec_loss_hq = rec_loss[:bs].mean()
                rec_loss_lq = rec_loss[bs:].mean()
                
        else:
            nll_loss = rec_loss
            if weights is not None:
                nll_loss = nll_loss * weights
            nll_loss = nll_loss.mean()
            with torch.no_grad():
                bs = inputs_hq.shape[0]
                rec_loss_hq = rec_loss[:bs].mean()
                rec_loss_lq = rec_loss[bs:].mean()

        kl_loss = post_hq.kl().mean() + post_lq.kl().mean()

        
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(rec_hq.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((rec_hq.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)
            
            '''
            contra_loss = torch.tensor(0.0, device=device)
            
            # Only compute if we have augmented features (training time)
            if self.contrastive_weight > 0 and feat_lq_aug is not None:
                # Anchor: feat_lq
                # Positive: feat_lq_aug
                # Negative: feat_hq (and potentially other HQs in batch)

                # Similarity (Anchor vs Positive) -> (B, 1)
                # Elements are normalized, so dot product is cosine similarity
                pos_sim = torch.sum(feat_lq * feat_lq_aug, dim=1, keepdim=True)
                
                # Similarity (Anchor vs Negatives) -> (B, B)
                # We treat all HQ images in the batch as negatives for the LQ images
                neg_sim = torch.matmul(feat_lq, feat_hq.T)
                
                # Logits: Concatenate Positive and Negatives
                # Shape: (B, 1 + B)
                logits = torch.cat([pos_sim, neg_sim], dim=1)
                logits = logits / self.contrastive_temp
                
                # Target: The positive is always at index 0
                labels_contrastive = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
                
                contra_loss = F.cross_entropy(logits, labels_contrastive)
            '''
            cls_loss = torch.tensor(0.0, device=device)
            if self.cls_weight > 0:
                hq_targets = torch.full_like(labels, 10)
                all_logits = torch.cat([logits_lq, logits_hq], dim=0)
                all_targets = torch.cat([labels, hq_targets], dim=0)
                cls_loss = self.classification_loss(all_logits, all_targets)

            # Adaptive GAN weight scheduling
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            
            # Total Loss
            loss = nll_loss + \
                   self.kl_weight * kl_loss + \
                   d_weight * disc_factor * g_loss + \
                   self.cls_weight * cls_loss
                   #self.contrastive_weight * contra_loss
            
            log = {
                "{}/total_loss".format(split): loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(),
                "{}/rec_loss_hq".format(split): rec_loss_hq.detach().mean(),
                "{}/rec_loss_lq".format(split): rec_loss_lq.detach().mean(),
                #"{}/contra_loss".format(split): contra_loss.detach().mean(),
                "{}/cls_loss".format(split): cls_loss.detach().mean(),
                "{}/g_loss".format(split): g_loss.detach().mean(),
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/logvar".format(split): self.logvar.detach(),
            }
            return loss, log

        if optimizer_idx == 1:
            if cond is None:
                logits_real = self.discriminator(inputs_hq.contiguous().detach())
                logits_fake = self.discriminator(rec_hq.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs_hq.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((rec_hq.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean()
            }
            return d_loss, log