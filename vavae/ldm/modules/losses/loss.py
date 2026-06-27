import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from torchvision import models
from taming.modules.losses.vqperceptual import * 

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        # vgg_outputs = namedtuple("VggOutputs", ['relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        # out = vgg_outputs(h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out

def gaussian_map(h, w, mu=0, sigma=1):
    x = torch.linspace(-1, 1, h)
    y = torch.linspace(-1, 1, w)
    x, y = torch.meshgrid(x, y)
    gauss = 1 / (2 * math.pi * sigma ** 2) * torch.exp(-((x - mu) ** 2 + (y - mu) ** 2) / (2 * sigma ** 2))
    return gauss.unsqueeze(0).unsqueeze(0)

def normalize_batch(batch, final_func='sig'):
    # normalize using imagenet mean and std
    if final_func == 'tanh':
        batch = (batch + 1.0) / 2.0
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    return (batch - mean) / std

def Gram_matrix(input):
    a,b,c,d = input.size()
    features = input.view(a*b, c*d)
    G = torch.mm(features, features.t())

    return G.div(a*b*c*d)

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg16(requires_grad=False)
        self.vgg.eval()

    def forward(self, pred, target):
        loss_perc = 0
        # print(pred.shape, target.shape)
        predict_features = self.vgg(normalize_batch(pred))
        target_features = self.vgg(normalize_batch(target))
        for f_x, f_y in zip(predict_features, target_features):
            loss_perc += torch.mean((f_x - f_y)**2)
            G_x = Gram_matrix(f_x)
            G_y = Gram_matrix(f_y)
            loss_perc += torch.mean((G_x - G_y)**2)
        return loss_perc



def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss
    
class GANLoss(nn.Module):
    def __init__(self, disc_in_channels=3, disc_num_layers=3, disc_loss="hinge"):
        super(GANLoss, self).__init__()
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=False
                                                 ).apply(weights_init)
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss


    def forward(self, pred, target, pred_idx=0):
        if pred_idx == 0:
            logits_real = self.discriminator(target.contiguous().detach())
            logits_fake = self.discriminator(pred.contiguous().detach())
            return self.disc_loss(logits_real, logits_fake)
        elif pred_idx == 1:
            logits_fake = self.discriminator(pred.contiguous().detach())
            return -torch.mean(logits_fake)