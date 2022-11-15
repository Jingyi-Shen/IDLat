import math

from numpy.testing._private.utils import measure

import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim as MS_SSIM
import torch.nn.functional as F

import sys 
sys.path.append("..")
from dataset import denormalize_zscore, denormalize_max_min

class PixelwiseRateDistortionLoss(nn.Module):
    def __init__(self, beta=10, block_size=16, dataname='vortex'):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.l1 = nn.L1Loss(reduction='none')
        self.beta = beta
        self.block_size = block_size
        self.cos_sim =  F.cosine_similarity
        self.dataname = dataname
        
    def forward(self, output, target, lmbdamap):
        # lmbdamap: (B, 1, H, W)
        N, C, _, _, _ = target.size() # channels
        num_voxels = N * C * self.block_size * self.block_size * self.block_size
        # num_voxels = N * Z * Y * X
        out = {}

        out['bpp_loss'] = sum(
            (-torch.log2(likelihoods).sum() / num_voxels)
            for likelihoods in output['likelihoods'].values()
        )

        
        # if self.dataname == 'tornado' or self.dataname == 'nyx':
        if self.dataname == 'tornado':
            mse = self.l1(output['x_hat'], target)
        else:
            mse = self.mse(output['x_hat'], target)
        # import pdb
        # pdb.set_trace()
        
        lmbdamap = lmbdamap.expand_as(mse)
        out['mse_loss'] = torch.mean(lmbdamap * mse)
        out['loss'] = self.beta * out['mse_loss'] + out['bpp_loss']
        
        if self.dataname == 'tornado':
            cos_loss = 1 - self.cos_sim(output['x_hat'], target, dim=1)
            out['loss'] = out['loss'] + torch.mean(lmbdamap * torch.unsqueeze(cos_loss, dim=1))
            
        return out


class DistortionLoss(nn.Module):
    def __init__(self, beta=10, block_size=16, dataname='vortex'):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.l1 = nn.L1Loss(reduction='none')
        self.beta = beta
        self.block_size = block_size
        self.cos_sim =  F.cosine_similarity
        self.dataname = dataname
        
    def forward(self, output, target, lmbdamap):
        # lmbdamap: (B, 1, H, W)
        out = {}
        # if self.dataname == 'tornado' or self.dataname == 'nyx':
        if self.dataname == 'tornado':
            mse = self.l1(output['x_hat'], target)
        else:
            mse = self.mse(output['x_hat'], target)
        # import pdb
        # pdb.set_trace()
        
        lmbdamap = lmbdamap.expand_as(mse)
        out['mse_loss'] = torch.mean(lmbdamap * mse)
        out['loss'] = self.beta * out['mse_loss']
        
        out['bpp_loss'] = out['mse_loss'] # fake data, do not count bpp loss 
        
        if self.dataname == 'tornado':
            cos_loss = 1 - self.cos_sim(output['x_hat'], target, dim=1)
            out['loss'] = out['loss'] + torch.mean(lmbdamap * torch.unsqueeze(cos_loss, dim=1))
            
        return out


class LatentLerp_DistortionLoss(nn.Module):
    def __init__(self, beta=10, block_size=16, dataname='vortex'):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.l1 = nn.L1Loss(reduction='none')
        self.beta = beta
        self.block_size = block_size
        self.dataname = dataname
        
    def forward(self, output, target, lmbdamap):
        # lmbdamap: (B, 1, H, W)
        N, C, _, _, _ = target.size() # channels
        num_voxels = N * C * self.block_size * self.block_size * self.block_size
        # num_voxels = N * Z * Y * X
        out = {}

        out['bpp_loss'] = sum(
            (-torch.log2(likelihoods).sum() / num_voxels)
            for likelihoods in output['likelihoods'].values()
        )
        
        # if self.dataname == 'tornado' or self.dataname == 'nyx':
        if self.dataname == 'tornado':
            mse = self.l1(output['x_hat'], target)
        else:
            mse = self.mse(output['x_hat'], target)
        # import pdb
        # pdb.set_trace()
        
        lmbdamap = lmbdamap.expand_as(mse)
        out['mse_loss'] = torch.mean(lmbdamap * mse)
        out['latent_loss'] = 
        out['loss'] = self.beta * out['mse_loss'] + out['bpp_loss']
        
        if self.dataname == 'tornado':
            cos_loss = 1 - self.cos_sim(output['x_hat'], target, dim=1)
            out['loss'] = out['loss'] + torch.mean(lmbdamap * torch.unsqueeze(cos_loss, dim=1))
            
        return out



class LatentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        
    def forward(self, out_forward, out_backward, out_cur, start_lat, end_lat):
        # l1_forward = self.l1(out_forward, end_lat)
        # l1_backward = self.l1(out_backward, start_lat)
        # l1_cur = self.l1(out_cur, start_lat)
        out = {}
        out['loss'] = torch.mean(self.l1(out_forward, end_lat) + self.l1(out_backward, start_lat) + self.l1(out_cur, start_lat)) 
        return out


class Metrics(nn.Module):
    def __init__(self, norm_method='mean_std', mean_std_norm=None, min_max_norm=None, block_size=16, Q=False):
        super().__init__()
        self.diff = min_max_norm['max']-min_max_norm['min']
        self.mean_std_norm = mean_std_norm
        self.min_max_norm = min_max_norm
        self.norm_method = norm_method
        self.block_size = block_size
        self.Q = Q
        
    def MSE(self, x, y, qmap=None):
        # MSE or Weighted Mean Squared Error (WMSE)
        # x, y: 5D # 4D [0, 1]
        if self.norm_method == 'mean_std':
            x = denormalize_zscore(x, self.mean_std_norm)
            y = denormalize_zscore(y, self.mean_std_norm)
        elif self.norm_method == 'min_max':
            x = denormalize_max_min(x, self.min_max_norm)
            y = denormalize_max_min(y, self.min_max_norm)
        # if qmap is not None:
        #     print('before', qmap.shape)
        #     qmap = qmap.expand_as(x)
        #     print('after', qmap.shape)
        #     return torch.mean(qmap * (x - y) ** 2, dim=[1, 2, 3, 4])
        # out['mse_loss'] = torch.mean(lmbdamap * mse)
        return torch.mean((x - y) ** 2, dim=[1, 2, 3, 4])

    def PSNR(self, x, y):
        # x, y: 5D [0, 1]
        # import pdb
        # pdb.set_trace()
        mse = self.MSE(x, y)
        psnr = 10 * torch.log10(self.diff ** 2. / mse)  # (B,)
        return torch.mean(psnr)

    def forward(self, output, target):
        
        if self.Q: 
            psnr = self.PSNR(output['x_hat'], target)
            return psnr, psnr
        else:
            N, C, Z, Y, X = target.size()
            
            num_voxels = N * C * self.block_size * self.block_size * self.block_size
    
            bpp = sum(
                (-torch.log2(likelihoods).sum() / num_voxels)
                for likelihoods in output['likelihoods'].values()
            )
            psnr = self.PSNR(output['x_hat'], target)
            # ms_ssim = self.MS_SSIM(output['x_hat'], target)
            return bpp, psnr
