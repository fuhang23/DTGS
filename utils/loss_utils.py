#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
#
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def TV_loss(x, mask):
    B, C, H, W = x.shape
    tv_h = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).sum()
    tv_w = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).sum()
    return (tv_h + tv_w) / (B * C * H * W)


def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()

def l1_loss(network_output, gt, mask=None):
    loss = torch.abs((network_output - gt))
    if mask is not None:
        if mask.ndim == 4:
            mask = mask.repeat(1, network_output.shape[1], 1, 1)
        elif mask.ndim == 3:
            mask = mask.repeat(network_output.shape[1], 1, 1)
        else:
            raise ValueError('the dimension of mask should be either 3 or 4')
    
        try:
            loss = loss[mask!=0]
        except:
            print(loss.shape)
            print(mask.shape)
            print(loss.dtype)
            print(mask.dtype)
    return loss.mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    
    img1 = F.interpolate(img1, scale_factor=0.5, mode='bilinear', align_corners=False)
    img2 = F.interpolate(img2, scale_factor=0.5, mode='bilinear', align_corners=False)
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class MonoSSIMLoss(torch.nn.Module):
    def __init__(self, window_size=5, edge_weight=2.0):
        super(MonoSSIMLoss, self).__init__()
        self.window_size = window_size  
        self.edge_weight = edge_weight  
        self.C1 = (0.01 * 255) ** 2  
        self.C2 = (0.03 * 255) ** 2  

    def _gaussian_kernel(self, channel=1):
        """Generate a Gaussian kernel for local statistics."""
        kernel = torch.tensor([[1, 4, 6, 4, 1],
                               [4, 16, 24, 16, 4],
                               [6, 24, 36, 24, 6],
                               [4, 16, 24, 16, 4],
                               [1, 4, 6, 4, 1]], dtype=torch.float32)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, self.window_size, self.window_size)
        kernel = kernel.repeat(channel, 1, 1, 1)
        return kernel.to(next(self.parameters()).device)

    def _calculate_edge_weight(self, img):
        """Compute edge weights from image gradients."""
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(img.shape[1], 1, 1, 1).to(img.device)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(img.shape[1], 1, 1, 1).to(img.device)

        grad_x = F.conv2d(img, sobel_x, padding=1, groups=img.shape[1])
        grad_y = F.conv2d(img, sobel_y, padding=1, groups=img.shape[1])
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        
        grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
        edge_weight = 1 + self.edge_weight * grad_mag
        return edge_weight

    def forward(self, pred, target):
        """Compute the MonoSSIM loss."""
        B, C, H, W = pred.shape
        kernel = self._gaussian_kernel(C)

        
        mu_pred = F.conv2d(pred, kernel, padding=self.window_size // 2, groups=C)
        mu_target = F.conv2d(target, kernel, padding=self.window_size // 2, groups=C)

        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred = F.conv2d(pred * pred, kernel, padding=self.window_size // 2, groups=C) - mu_pred_sq
        sigma_target = F.conv2d(target * target, kernel, padding=self.window_size // 2, groups=C) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, kernel, padding=self.window_size // 2, groups=C) - mu_pred_target

        
        ssim_numerator = (2 * mu_pred_target + self.C1) * (2 * sigma_pred_target + self.C2)
        ssim_denominator = (mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred + sigma_target + self.C2)
        ssim = ssim_numerator / (ssim_denominator + 1e-8)

        
        edge_weight = self._calculate_edge_weight(target)  
        weighted_ssim = ssim * edge_weight

        
        loss = 1 - torch.mean(weighted_ssim)
        return loss

def ssim_train(img1, img2, mask, window_size=11, size_average=True):
    
    img1 = img1 * mask
    img2 = img2 * mask
    
    img1 = F.interpolate(img1, scale_factor=0.5, mode='bilinear', align_corners=False)
    img2 = F.interpolate(img2, scale_factor=0.5, mode='bilinear', align_corners=False)
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
    # return _tssim(img1, img2, window, window_size, channel, size_average)

def _format_mask(mask, img):
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError('the dimension of mask should be either 3 or 4')
    if mask.shape[1] == 1 and img.shape[1] != 1:
        mask = mask.repeat(1, img.shape[1], 1, 1)
    return mask.to(device=img.device, dtype=img.dtype)

def tssim_train(img1, img2, mask=None, window_size=11, size_average=True, eps=1e-8):
    # DTGS Eq. (14)(15): use GT thermal radiance as the weight for mean,
    # variance, and covariance statistics.
    channel = img1.size(-3)
    window = create_window(window_size, channel).to(device=img1.device, dtype=img1.dtype)
    mask = _format_mask(mask, img1)
    weight = img2.clamp_min(0.0)
    if mask is not None:
        weight = weight * mask

    weight_sum = F.conv2d(weight, window, padding=window_size // 2, groups=channel).clamp_min(eps)
    mu1 = F.conv2d(img1 * weight, window, padding=window_size // 2, groups=channel) / weight_sum
    mu2 = F.conv2d(img2 * weight, window, padding=window_size // 2, groups=channel) / weight_sum

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    sigma1_sq = F.conv2d(img1 * img1 * weight, window, padding=window_size // 2, groups=channel) / weight_sum - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2 * weight, window, padding=window_size // 2, groups=channel) / weight_sum - mu2_sq
    sigma12 = F.conv2d(img1 * img2 * weight, window, padding=window_size // 2, groups=channel) / weight_sum - mu1 * mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    tssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps)

    if mask is not None:
        valid = F.conv2d(mask, window, padding=window_size // 2, groups=channel) > 0
        if not valid.any():
            return tssim_map.sum()
        tssim_map = tssim_map[valid]

    if size_average:
        return tssim_map.mean()
    return tssim_map.mean(1).mean(1).mean(1)

def tssim_loss(img1, img2, mask=None, window_size=11):
    return 1 - tssim_train(img1, img2, mask, window_size=window_size)

# Backward-compatible aliases for earlier experiments.
hssim_train = tssim_train
hssim_loss = tssim_loss

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def _tssim(img1, img2, window, window_size, channel, size_average=True, C1=0.01**2, C2=0.03**2):
    return tssim_train(img1, img2, window_size=window_size, size_average=size_average)

