import torch
import torch.nn as nn
import torch.nn.functional as F
from taming.modules.losses.vqperceptual import LPIPS, NLayerDiscriminator, weights_init, hinge_d_loss, vanilla_d_loss, adopt_weight

class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=0.1,
                 perceptual_weight=1.2, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", pp_style=False, vf_weight=1e2, adaptive_vf=False,
                 cos_margin=0, distmat_margin=0, distmat_weight=1.0, cos_weight=1.0,
                 **kwargs # Absorb unused arguments from old configs
                 ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.distmat_weight = distmat_weight
        self.cos_weight = cos_weight
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
        self.nll_weights = 1.0
        self.cos_margin = cos_margin
        self.distmat_margin = distmat_margin
        
        

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.06, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    

    def forward(self, inputs,rec,post,labels,
                optimizer_idx, global_step,
                last_layer=None, cond=None, split="train",
                weights=None, enc_last_layer=None,
                **kwargs):
        
        device = inputs.device
        #rec loss
        rec_loss = torch.abs(inputs.contiguous() - rec.contiguous())
        p_loss = torch.tensor(0.0, device=device)
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs, rec)
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        # nll loss
        if not self.pp_style:
            nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar          
        else:
            nll_loss = rec_loss
        nll_loss = self.nll_weights * nll_loss  
        nll_loss = nll_loss.mean()
        # kl loss
        kl_loss = post.kl().mean()

        if optimizer_idx == 0:
            # generator update
            g_loss = torch.tensor(0.0).to(inputs.device)
            d_weight = torch.tensor(0.0)
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(rec.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((rec.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            # Adaptive GAN weight scheduling
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            # Total Loss
            loss = nll_loss + \
                self.kl_weight * kl_loss + \
                d_weight * disc_factor * g_loss 

                  
            log = {
                "{}/total_loss".format(split): loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/g_loss".format(split): g_loss.detach().mean(),
                "{}/p_loss".format(split): p_loss.detach().mean(),
                "{}/d_weight".format(split): d_weight.detach(),
            }
            return loss, log

        if optimizer_idx == 1:
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(rec.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((rec.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean()
            }
            return d_loss, log