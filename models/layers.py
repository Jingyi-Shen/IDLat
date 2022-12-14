import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.parametrizers import NonNegativeParametrizer


def Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding_mode='reflect'):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride,
                     padding=kernel_size // 2, padding_mode=padding_mode)


def UpConv2d(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

def Conv3d(in_channels, out_channels, kernel_size=5, stride=2, padding_mode='replicate'):
    return nn.Conv3d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride,
                     padding=kernel_size // 2, padding_mode=padding_mode)


def UpConv3d(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose3d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

class SFT(nn.Module):
    def __init__(self, x_nc, prior_nc=1, ks=3, nhidden=32):
        super().__init__()
        pw = ks // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv3d(prior_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
            # nn.LeakyReLU(0.1, True),
        )
        self.mlp_gamma = nn.Conv3d(nhidden, x_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv3d(nhidden, x_nc, kernel_size=ks, padding=pw)

    def forward(self, x, qmap):
        qmap = F.adaptive_avg_pool3d(qmap, x.size()[2:]) 
        actv = self.mlp_shared(qmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = x * (1 + gamma)  + beta
        # s = x.size(dim=2)
        # import numpy as np
        # np.save('./SFT/lighter_gamma_%d.npy'%s, gamma.detach().cpu().numpy())
        # np.save('./SFT/lighter_beta_%d.npy'%s, beta.detach().cpu().numpy())
        # np.save('./SFT/lighter_x_%d.npy'%s, x.detach().cpu().numpy())
        # np.save('./SFT/lighter_qmap_%d.npy'%s, qmap.detach().cpu().numpy())
        # print(x.shape, gamma.shape, beta.shape, s)
        return out


class SFTResblk(nn.Module):
    def __init__(self, x_nc, prior_nc, ks=3):
        super().__init__()
        self.conv_0 = nn.Conv3d(x_nc, x_nc, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv3d(x_nc, x_nc, kernel_size=3, padding=1)

        self.norm_0 = SFT(x_nc, prior_nc, ks=ks)
        self.norm_1 = SFT(x_nc, prior_nc, ks=ks)

    def forward(self, x, qmap):
        dx = self.conv_0(self.actvn(self.norm_0(x, qmap)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, qmap)))
        out = x + dx

        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """
    def __init__(self, in_channels, inverse=False, beta_min=1e-6, gamma_init=0.1):  # beta_min=1e-6
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x):
        _, C, _, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1, 1)
        norm = F.conv3d(x ** 2, gamma, beta)

        if self.inverse:
            norm = torch.sqrt(norm)  # may cause nan.
            # norm = torch.sqrt(torch.relu(norm))
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out


class GDN1(GDN):
    def forward(self, x):
        _, C, _, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1, 1)
        norm = F.conv3d(torch.abs(x), gamma, beta)

        if not self.inverse:
            norm = 1.0 / (norm + 1e-4)

        out = x * norm

        return out