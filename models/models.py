import imp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .hyperpriors import ScaleHyperprior
from .utils import conv
from .layers import GDN1

from .layers import SFT, SFTResblk, Conv3d, UpConv3d

import pdb
import time

class MySpatiallyAdaptiveCompression_lighter(ScaleHyperprior):
    def __init__(self, N=16, M=16, sft_ks=3, prior_nc=16, inmap=1, ind=1, **kwargs):
        super().__init__(N, M, **kwargs)
        ### condition networks ###
        # g_a,c
        self.inmap = inmap
        self.ind = ind
        self.qmap_feature_g1 = nn.Sequential(
            conv(inmap, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            # conv(prior_nc * 4, prior_nc * 2, 3, 1),
            # nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_g2 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g3 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g4 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )

        # g_s,c
        self.qmap_feature_gs1 = nn.Sequential(
            conv(M, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            # conv(prior_nc * 4, prior_nc * 2, 3, 1),
            # nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_gs2 = nn.Sequential(
            UpConv3d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_gs3 = nn.Sequential(
            UpConv3d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_gs4 = nn.Sequential(
            UpConv3d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        
        ### compression networks ###
        # g_a, encoder
        self.g_a = None
        self.g_a0 = Conv3d(self.ind, N//4, kernel_size=5, stride=1)
        self.g_a1 = GDN1(N//4)
        self.g_a2 = SFT(N // 4, prior_nc, ks=sft_ks) # output C: N // 4

        self.g_a3 = Conv3d(N//4, N//2)
        self.g_a4 = GDN1(N//2)
        self.g_a5 = SFT(N // 2, prior_nc, ks=sft_ks)

        self.g_a6 = Conv3d(N//2, N)
        self.g_a7 = GDN1(N)
        self.g_a8 = SFT(N, prior_nc, ks=sft_ks)
        
        self.g_a9 = Conv3d(N, M)
        
        # gs
        self.g_s = None
        self.g_s1 = UpConv3d(M, N)
        self.g_s2 = GDN1(N, inverse=True)
        self.g_s3 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s4 = UpConv3d(N, N)
        self.g_s5 = GDN1(N, inverse=True)
        self.g_s6 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s7 = UpConv3d(N, N // 2)
        self.g_s8 = GDN1(N // 2, inverse=True)
        self.g_s9 = SFT(N // 2, prior_nc, ks=sft_ks)
        
        self.g_s10 = Conv3d(N // 2, self.ind, kernel_size=5, stride=1)

    def g_a(self, x, qmap):
        qmap = self.qmap_feature_g1(qmap)
        x = self.g_a0(x)
        x = self.g_a1(x)
        x = self.g_a2(x, qmap)
        
        qmap = self.qmap_feature_g2(qmap)
        x = self.g_a3(x)
        x = self.g_a4(x)
        x = self.g_a5(x, qmap)

        qmap = self.qmap_feature_g3(qmap)
        x = self.g_a6(x)
        x = self.g_a7(x)
        x = self.g_a8(x, qmap)

        qmap = self.qmap_feature_g4(qmap)
        x = self.g_a9(x)
        
        return x

    def g_s(self, x):
        # gs
        w = self.qmap_feature_gs1(x)
        x = self.g_s1(x)
        x = self.g_s2(x)
        x = self.g_s3(x, w)
        

        w = self.qmap_feature_gs2(w)
        x = self.g_s4(x)
        x = self.g_s5(x)
        x = self.g_s6(x, w)

        w = self.qmap_feature_gs3(w)
        x = self.g_s7(x)
        x = self.g_s8(x)
        x = self.g_s9(x, w)

        x = self.g_s10(x)

        return x

    def forward(self, x, qmap):
        y = self.g_a(x, qmap) # 3, 3, 3
        # import pdb
        # pdb.set_trace()
        
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods},
        }

    def compress(self, x, qmap):
        y = self.g_a(x, qmap) # whole: [1, 16, 16, 16, 16]; [B, 16, 3, 3, 3]
        # import pdb
        # pdb.set_trace()
        y_strings, _ = self.entropy_bottleneck.compress(y)
        return {"strings": y_strings, "shape": y.size()[-3:]}

    def decompress(self, strings, shape):
        # assert isinstance(strings, list) and len(strings) == 2
        y_hat = self.entropy_bottleneck.decompress(strings, shape)
        x_hat = self.g_s(y_hat)#.clamp_(0, 1)
        return {"x_hat": x_hat, "y_hat": y_hat}

    def decode(self, y_hat):
        x_hat = self.g_s(y_hat)
        return x_hat

    def encode_decode_noquantize(self, x, qmap):
        enc_start = time.time()
        y = self.g_a(x, qmap) # whole: [1, 16, 16, 16, 16]; [B, 16, 3, 3, 3]
        enc_time = time.time() - enc_start
        dec_start = time.time()
        x_hat = self.g_s(y)
        dec_time = time.time() - dec_start
        return {"x_hat": x_hat, "y":y}, enc_time, dec_time


class MySpatiallyAdaptiveCompression(ScaleHyperprior):
    def __init__(self, N=16, M=16, sft_ks=3, prior_nc=16, inmap=1, ind=1, **kwargs):
        super().__init__(N, M, **kwargs)
        ### condition networks ###
        # g_a,c
        self.inmap = inmap
        self.ind = ind
        self.qmap_feature_g1 = nn.Sequential(
            conv(inmap, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_g2 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g3 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g4 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        # self.qmap_feature_g5 = nn.Sequential(
        #     conv(prior_nc, prior_nc, 3),
        #     nn.LeakyReLU(0.1, True),
        #     conv(prior_nc, prior_nc, 1, 1)
        # )
 
        # h_a,c
        self.qmap_feature_h1 = nn.Sequential(
            conv(M + 1, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            # conv(prior_nc * 4, prior_nc * 2, 3, 1),
            # nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_h2 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )

        # f_c
        self.qmap_feature_gs0 = nn.Sequential(
            # nn.ConvTranspose3d(N, N//2, kernel_size=2, stride=1, padding=0), # stride=1 for 3->4, stride=2 for x2;
            nn.ConvTranspose3d(N, N//2, kernel_size=3, stride=2, padding=1),# 2*in-1
            nn.LeakyReLU(0.1, True),
            # UpConv3d(N//2, N//4),
            # nn.LeakyReLU(0.1, True),
            conv(N//2, N//4, 3, 1)
        )

        # g_s,c
        self.qmap_feature_gs1 = nn.Sequential(
            conv(M + N // 4, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_gs2 = nn.Sequential(
            UpConv3d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_gs3 = nn.Sequential(
            UpConv3d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_gs4 = nn.Sequential(
            UpConv3d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )

        ### compression networks ###
        # g_a, encoder
        self.g_a = None
        self.g_a0 = Conv3d(self.ind, N//4, kernel_size=5, stride=1)
        self.g_a1 = GDN1(N//4)
        self.g_a2 = SFT(N // 4, prior_nc, ks=sft_ks) # output C: N // 4

        self.g_a3 = Conv3d(N//4, N//2)
        self.g_a4 = GDN1(N//2)
        self.g_a5 = SFT(N // 2, prior_nc, ks=sft_ks)

        self.g_a6 = Conv3d(N//2, N)
        self.g_a7 = GDN1(N)
        self.g_a8 = SFT(N, prior_nc, ks=sft_ks)

        self.g_a12 = Conv3d(N, M)
        self.g_a13 = SFTResblk(M, prior_nc, ks=sft_ks)
        self.g_a14 = SFTResblk(M, prior_nc, ks=sft_ks)

        # h_a, hyper encoder
        self.h_a = None
        self.h_a0 = Conv3d(M, N, kernel_size=3, stride=1)
        self.h_a1 = SFT(N, prior_nc, ks=sft_ks)
        self.h_a2 = nn.LeakyReLU(inplace=True)
        
        self.h_a3 = Conv3d(N, N)
        self.h_a4 = SFTResblk(N, prior_nc, ks=sft_ks)
        self.h_a5 = SFTResblk(N, prior_nc, ks=sft_ks)

        # g_s, decoder
        self.g_s = None
        self.g_s0 = SFTResblk(M, prior_nc, ks=sft_ks)
        self.g_s1 = SFTResblk(M, prior_nc, ks=sft_ks)

        self.g_s2 = UpConv3d(M, N)
        self.g_s3 = GDN1(N, inverse=True)
        self.g_s4 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s5 = UpConv3d(N, N)
        self.g_s6 = GDN1(N, inverse=True)
        self.g_s7 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s8 = UpConv3d(N, N // 2)
        self.g_s9 = GDN1(N // 2, inverse=True)
        self.g_s10 = SFT(N // 2, prior_nc, ks=sft_ks)

        # self.g_s11 = UpConv3d(N // 2, N // 4)
        # self.g_s12 = GDN1(N // 4, inverse=True)
        # self.g_s13 = SFT(N // 4, prior_nc, ks=sft_ks)

        self.g_s14 = Conv3d(N // 2, self.ind, kernel_size=5, stride=1)

        # h_s, hyper decoder
        self.h_s = nn.Sequential(
            # nn.Upsample(size=3, mode='trilinear', align_corners=False),
            nn.ConvTranspose3d(N, M, kernel_size=2, stride=1, padding=0), # make (2,2,2) to (3,3,3)
            # nn.ConvTranspose3d(N, M, kernel_size=2, stride=2, padding=0),# suppose to X2;
            # nn.ConvTranspose3d(N, M, kernel_size=3, stride=2, padding=1),# 2*in-1
            nn.LeakyReLU(inplace=True),
            conv(M, M * 2, stride=1, kernel_size=3), # padding=kernel_size // 2, M * 2: mu and sigma 
        )

    def g_a(self, x, qmap):
        qmap = self.qmap_feature_g1(qmap)
        x = self.g_a0(x)
        x = self.g_a1(x)
        x = self.g_a2(x, qmap)

        qmap = self.qmap_feature_g2(qmap)
        x = self.g_a3(x)
        x = self.g_a4(x)
        x = self.g_a5(x, qmap)

        qmap = self.qmap_feature_g3(qmap)
        x = self.g_a6(x)
        x = self.g_a7(x)
        x = self.g_a8(x, qmap)

        qmap = self.qmap_feature_g4(qmap)
        x = self.g_a12(x)
        x = self.g_a13(x, qmap)
        x = self.g_a14(x, qmap)

        # qmap = self.qmap_feature_g5(qmap)
        # x = self.g_a12(x)
        # x = self.g_a13(x, qmap)
        # x = self.g_a14(x, qmap)
        return x

    def h_a(self, x, qmap):
        # pooling qmap into [3,3,3]
        qmap = F.adaptive_avg_pool3d(qmap, x.size()[2:])
        qmap = self.qmap_feature_h1(torch.cat([qmap, x], dim=1))
        x = self.h_a0(x)
        x = self.h_a1(x, qmap)
        x = self.h_a2(x)

        qmap = self.qmap_feature_h2(qmap)
        x = self.h_a3(x)
        x = self.h_a4(x, qmap)
        x = self.h_a5(x, qmap) #16, 2, 2, 2

        # qmap = self.qmap_feature_h3(qmap)
        # x = self.h_a6(x)
        # x = self.h_a7(x, qmap)
        # x = self.h_a8(x, qmap)
        return x

    def g_s(self, x, z):
        w = self.qmap_feature_gs0(z)
        w = self.qmap_feature_gs1(torch.cat([w, x], dim=1))
        x = self.g_s0(x, w)
        x = self.g_s1(x, w)

        w = self.qmap_feature_gs2(w)
        x = self.g_s2(x)
        x = self.g_s3(x)
        x = self.g_s4(x, w)

        w = self.qmap_feature_gs3(w)
        x = self.g_s5(x)
        x = self.g_s6(x)
        x = self.g_s7(x, w)

        w = self.qmap_feature_gs4(w)
        x = self.g_s8(x)
        x = self.g_s9(x)
        x = self.g_s10(x, w)

        # w = self.qmap_feature_gs5(w)
        # x = self.g_s11(x)
        # x = self.g_s12(x)
        # x = self.g_s13(x, w)

        x = self.g_s14(x)
        return x

    def forward(self, x, qmap):
        y = self.g_a(x, qmap) # 3, 3, 3
        z = self.h_a(y, qmap) # 2, 2, 2
        # import pdb
        # pdb.set_trace()
        
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(chunks=2, dim=1) # split a tensor into 2 chunks
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat, z_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, qmap):
        y = self.g_a(x, qmap) # whole: [1, 16, 16, 16, 16]; [B, 16, 3, 3, 3]
        z = self.h_a(y, qmap) # whole: [1, 16, 8, 8, 8]; [B, 16, 2, 2, 2]
        # import pdb
        # pdb.set_trace()
        # print(y.shape, z.shape)
        
        z_strings, z_symbols = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-3:]) # [B, 16, 2, 2, 2]
        # z_strings, _ = self.entropy_bottleneck.torchac_compress(z)
        # z_hat = self.entropy_bottleneck.torchac_decompress(z_strings, z.size()[-3:])
        gaussian_params = self.h_s(z_hat) # [B, 16, 2, 2, 2] --> [B, 16, 3, 3, 3]
        scales_hat, means_hat = gaussian_params.chunk(chunks=2, dim=1) # [B, 16, 3, 3, 3]
        # print(scales_hat.shape, means_hat.shape)
        indexes = self.gaussian_conditional.build_indexes(scales_hat) #  [B, 16, 3, 3, 3]
        y_strings, y_symbols = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-3:], "z_hat":z_hat} #, "y_symbols":y_symbols, "z_symbols":z_symbols}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        # z_hat = self.entropy_bottleneck.torchac_decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat, z_hat)#.clamp_(0, 1)
        return {"x_hat": x_hat, "y_hat": y_hat}

    def decode(self, y_hat, z_hat):
        x_hat = self.g_s(y_hat, z_hat)
        return x_hat

    def encode_decode_noquantize(self, x, qmap):
        enc_start = time.time()
        y = self.g_a(x, qmap) # whole: [1, 16, 16, 16, 16]; [B, 16, 3, 3, 3]
        z = self.h_a(y, qmap) # whole: [1, 16, 8, 8, 8]; [B, 16, 2, 2, 2]
        enc_time = time.time() - enc_start
        dec_start = time.time()
        x_hat = self.g_s(y, z)
        dec_time = time.time() - dec_start
        return {"x_hat": x_hat, "y":y, "z":z}, enc_time, dec_time
        
    def encoding_yz(self, x, qmap):
        enc_start = time.time()
        y = self.g_a(x, qmap)
        z = self.h_a(y, qmap)
        enc_time = time.time() - enc_start
        return {"y":y, "z":z}, enc_time

    # def get_z_hat_y_hat(self, x, qmap):
    #     y = self.g_a(x, qmap)
    #     z = self.h_a(y, qmap) # [128, 16, 2, 2, 2]
        
    #     z_strings = self.entropy_bottleneck.compress(z)
    #     z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-3:])
    #     gaussian_params = self.h_s(z_hat)
    #     scales_hat, means_hat = gaussian_params.chunk(2, 1)
    #     indexes = self.gaussian_conditional.build_indexes(scales_hat)
    #     y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
    #     y_hat = self.gaussian_conditional.decompress(
    #         y_strings, indexes, means=means_hat
    #     )
    #     return {"z_hat": z_hat, "y_hat":y_hat, "z_strings": z_strings, "y_strings":y_strings}
    
    # def get_x_hat(self, y_hat, z_hat):
    #     x_hat = self.g_s(y_hat, z_hat)
    #     return {"x_hat": x_hat}

class MySpatiallyAdaptiveCompression_AE(ScaleHyperprior):
    def __init__(self, N=16, M=16, sft_ks=3, prior_nc=16, inmap=1, ind=1, **kwargs):
        super().__init__(N, M, **kwargs)
        ### condition networks ###
        # g_a,c
        self.inmap = inmap
        self.ind = ind
        self.qmap_feature_g1 = nn.Sequential(
            conv(inmap, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_g2 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g3 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g4 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        
        # h_a,c
        self.qmap_feature_h1 = nn.Sequential(
            conv(M + 1, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            # conv(prior_nc * 4, prior_nc * 2, 3, 1),
            # nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_h2 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )

        # f_c
        self.qmap_feature_gs0 = nn.Sequential(
            # nn.ConvTranspose3d(N, N//2, kernel_size=2, stride=1, padding=0), # stride=1 for 3->4, stride=2 for x2;
            nn.ConvTranspose3d(N, N//2, kernel_size=3, stride=2, padding=1),# 2*in-1
            nn.LeakyReLU(0.1, True),
            # UpConv3d(N//2, N//4),
            # nn.LeakyReLU(0.1, True),
            # conv(N//2, N//4, 3, 1)
            conv(N//2, N//4, 3, 1)
        )

        # g_s,c
        self.qmap_feature_gs1 = nn.Sequential(
            conv(M + N // 4, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_gs2 = nn.Sequential(
            UpConv3d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_gs3 = nn.Sequential(
            UpConv3d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_gs4 = nn.Sequential(
            UpConv3d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )

        ### compression networks ###
        # g_a, encoder
        self.g_a = None
        self.g_a0 = Conv3d(self.ind, N//4, kernel_size=5, stride=1)
        self.g_a1 = GDN1(N//4)
        self.g_a2 = SFT(N // 4, prior_nc, ks=sft_ks) # output C: N // 4

        self.g_a3 = Conv3d(N//4, N//2)
        self.g_a4 = GDN1(N//2)
        self.g_a5 = SFT(N // 2, prior_nc, ks=sft_ks)

        self.g_a6 = Conv3d(N//2, N)
        self.g_a7 = GDN1(N)
        self.g_a8 = SFT(N, prior_nc, ks=sft_ks)

        self.g_a12 = Conv3d(N, M)
        self.g_a13 = SFTResblk(M, prior_nc, ks=sft_ks)
        self.g_a14 = SFTResblk(M, prior_nc, ks=sft_ks)

        # h_a, hyper encoder
        self.h_a = None
        self.h_a0 = Conv3d(M, N, kernel_size=3, stride=1)
        self.h_a1 = SFT(N, prior_nc, ks=sft_ks)
        self.h_a2 = nn.LeakyReLU(inplace=True)

        
        self.h_a3 = Conv3d(N, N)
        self.h_a4 = SFTResblk(N, prior_nc, ks=sft_ks)
        self.h_a5 = SFTResblk(N, prior_nc, ks=sft_ks)

        # g_s, decoder
        self.g_s = None
        self.g_s0 = SFTResblk(M, prior_nc, ks=sft_ks)
        self.g_s1 = SFTResblk(M, prior_nc, ks=sft_ks)

        # self.g_s2 = UpConv3d(M, N)
        self.g_s2 = UpConv3d(M + N // 4, N)
        self.g_s3 = GDN1(N, inverse=True)
        self.g_s4 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s5 = UpConv3d(N, N)
        self.g_s6 = GDN1(N, inverse=True)
        self.g_s7 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s8 = UpConv3d(N, N // 2)
        self.g_s9 = GDN1(N // 2, inverse=True)
        self.g_s10 = SFT(N // 2, prior_nc, ks=sft_ks)

        self.g_s14 = Conv3d(N // 2, self.ind, kernel_size=5, stride=1)

        # h_s, hyper decoder
        self.h_s = nn.Sequential(
            # nn.Upsample(size=3, mode='trilinear', align_corners=False),
            nn.ConvTranspose3d(N, M, kernel_size=2, stride=1, padding=0), # make (2,2,2) to (3,3,3)
            # nn.ConvTranspose3d(N, M, kernel_size=2, stride=2, padding=0),# suppose to X2;
            # nn.ConvTranspose3d(N, M, kernel_size=3, stride=2, padding=1),# 2*in-1
            nn.LeakyReLU(inplace=True),
            conv(M, M * 2, stride=1, kernel_size=3), # padding=kernel_size // 2, M * 2: mu and sigma 
        )

    def g_a(self, x, qmap):
        x = self.g_a0(x)
        x = self.g_a1(x)
     
        x = self.g_a3(x)
        x = self.g_a4(x)
        
        x = self.g_a6(x)
        x = self.g_a7(x)
        
        x = self.g_a12(x)
        return x

    def h_a(self, x, qmap):
        x = self.h_a0(x)
        x = self.h_a2(x)
        x = self.h_a3(x)
        return x

    def g_s(self, x, z):
        w = self.qmap_feature_gs0(z)
        x = self.g_s2(torch.cat([w, x], dim=1))
        x = self.g_s3(x)
        x = self.g_s5(x)
        x = self.g_s6(x)
        x = self.g_s8(x)
        x = self.g_s9(x)
        
        x = self.g_s14(x)
        return x

    def forward(self, x, qmap):
        y = self.g_a(x, qmap) # 3, 3, 3
        z = self.h_a(y, qmap) # 2, 2, 2
        # import pdb
        # pdb.set_trace()
        
        # z_hat, z_likelihoods = self.entropy_bottleneck(z)
        # gaussian_params = self.h_s(z_hat)
        # scales_hat, means_hat = gaussian_params.chunk(chunks=2, dim=1) # split a tensor into 2 chunks
        # y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y, z)

        return {
            "x_hat": x_hat,
        }

    def decode(self, y_hat, z_hat):
        x_hat = self.g_s(y_hat, z_hat)
        return x_hat

    def encode_decode_noquantize(self, x, qmap):
        enc_start = time.time()
        y = self.g_a(x, qmap) # whole: [1, 16, 16, 16, 16]; [B, 16, 3, 3, 3]
        z = self.h_a(y, qmap) # whole: [1, 16, 8, 8, 8]; [B, 16, 2, 2, 2]
        enc_time = time.time() - enc_start
        dec_start = time.time()
        x_hat = self.g_s(y, z)
        dec_time = time.time() - dec_start
        return {"x_hat": x_hat, "y":y, "z":z}, enc_time, dec_time



class MySpatiallyAdaptiveCompression_AE_IMP(ScaleHyperprior):
    def __init__(self, N=16, M=16, sft_ks=3, prior_nc=16, inmap=1, ind=1, **kwargs):
        super().__init__(N, M, **kwargs)
        ### condition networks ###
        # g_a,c
        self.inmap = inmap
        self.ind = ind
        self.qmap_feature_g1 = nn.Sequential(
            conv(inmap, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_g2 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g3 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g4 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
 
        # h_a,c
        self.qmap_feature_h1 = nn.Sequential(
            conv(M + 1, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            # conv(prior_nc * 4, prior_nc * 2, 3, 1),
            # nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_h2 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        
        # f_c
        self.qmap_feature_gs0 = nn.Sequential(
            # nn.ConvTranspose3d(N, N//2, kernel_size=2, stride=1, padding=0), # stride=1 for 3->4, stride=2 for x2;
            nn.ConvTranspose3d(N, N//2, kernel_size=3, stride=2, padding=1),# 2*in-1
            nn.LeakyReLU(0.1, True),
            # UpConv3d(N//2, N//4),
            # nn.LeakyReLU(0.1, True),
            # conv(N//2, N//4, 3, 1)
            conv(N//2, N//4, 3, 1)
        )

        # g_s,c
        self.qmap_feature_gs1 = nn.Sequential(
            conv(M + N // 4, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_gs2 = nn.Sequential(
            UpConv3d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_gs3 = nn.Sequential(
            UpConv3d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_gs4 = nn.Sequential(
            UpConv3d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )

        ### compression networks ###
        # g_a, encoder
        self.g_a = None
        self.g_a0 = Conv3d(self.ind, N//4, kernel_size=5, stride=1)
        self.g_a1 = GDN1(N//4)
        self.g_a2 = SFT(N // 4, prior_nc, ks=sft_ks) # output C: N // 4

        self.g_a3 = Conv3d(N//4, N//2)
        self.g_a4 = GDN1(N//2)
        self.g_a5 = SFT(N // 2, prior_nc, ks=sft_ks)

        self.g_a6 = Conv3d(N//2, N)
        self.g_a7 = GDN1(N)
        self.g_a8 = SFT(N, prior_nc, ks=sft_ks)

        self.g_a12 = Conv3d(N, M)
        self.g_a13 = SFTResblk(M, prior_nc, ks=sft_ks)
        self.g_a14 = SFTResblk(M, prior_nc, ks=sft_ks)

        # h_a, hyper encoder
        self.h_a = None
        self.h_a0 = Conv3d(M, N, kernel_size=3, stride=1)
        self.h_a1 = SFT(N, prior_nc, ks=sft_ks)
        self.h_a2 = nn.LeakyReLU(inplace=True)

        self.h_a3 = Conv3d(N, N)
        self.h_a4 = SFTResblk(N, prior_nc, ks=sft_ks)
        self.h_a5 = SFTResblk(N, prior_nc, ks=sft_ks)

        # g_s, decoder
        self.g_s = None
        self.g_s0 = SFTResblk(M, prior_nc, ks=sft_ks)
        self.g_s1 = SFTResblk(M, prior_nc, ks=sft_ks)

        # self.g_s2 = UpConv3d(M, N)
        self.g_s2 = UpConv3d(M + N // 4, N)
        self.g_s3 = GDN1(N, inverse=True)
        self.g_s4 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s5 = UpConv3d(N, N)
        self.g_s6 = GDN1(N, inverse=True)
        self.g_s7 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s8 = UpConv3d(N, N // 2)
        self.g_s9 = GDN1(N // 2, inverse=True)
        self.g_s10 = SFT(N // 2, prior_nc, ks=sft_ks)

        self.g_s14 = Conv3d(N // 2, self.ind, kernel_size=5, stride=1)

        # h_s, hyper decoder
        self.h_s = nn.Sequential(
            # nn.Upsample(size=3, mode='trilinear', align_corners=False),
            nn.ConvTranspose3d(N, M, kernel_size=2, stride=1, padding=0), # make (2,2,2) to (3,3,3)
            # nn.ConvTranspose3d(N, M, kernel_size=2, stride=2, padding=0),# suppose to X2;
            # nn.ConvTranspose3d(N, M, kernel_size=3, stride=2, padding=1),# 2*in-1
            nn.LeakyReLU(inplace=True),
            conv(M, M * 2, stride=1, kernel_size=3), # padding=kernel_size // 2, M * 2: mu and sigma 
        )

    def g_a(self, x, qmap):
        qmap = self.qmap_feature_g1(qmap)
        x = self.g_a0(x)
        x = self.g_a1(x)
        x = self.g_a2(x, qmap)

        qmap = self.qmap_feature_g2(qmap)
        x = self.g_a3(x)
        x = self.g_a4(x)
        x = self.g_a5(x, qmap)

        qmap = self.qmap_feature_g3(qmap)
        x = self.g_a6(x)
        x = self.g_a7(x)
        x = self.g_a8(x, qmap)

        qmap = self.qmap_feature_g4(qmap)
        x = self.g_a12(x)
        x = self.g_a13(x, qmap)
        x = self.g_a14(x, qmap)

        return x

    def h_a(self, x, qmap):
        # pooling qmap into [3,3,3]
        qmap = F.adaptive_avg_pool3d(qmap, x.size()[2:])
        qmap = self.qmap_feature_h1(torch.cat([qmap, x], dim=1))
        x = self.h_a0(x)
        x = self.h_a1(x, qmap)
        x = self.h_a2(x)

        x = self.h_a3(x)
        x = self.h_a4(x, qmap)
        x = self.h_a5(x, qmap) #16, 2, 2, 2

        return x

    def g_s(self, x, z):
        # import pdb
        # pdb.set_trace()
        w = self.qmap_feature_gs0(z)
        w = self.qmap_feature_gs1(torch.cat([w, x], dim=1))
        x = self.g_s0(x, w)
        x = self.g_s1(x, w)

        w = self.qmap_feature_gs2(w)
        x = self.g_s2(torch.cat([w, x], dim=1))
        x = self.g_s3(x)
        x = self.g_s4(x, w)

        w = self.qmap_feature_gs3(w)
        x = self.g_s5(x)
        x = self.g_s6(x)
        x = self.g_s7(x, w)

        w = self.qmap_feature_gs4(w)
        x = self.g_s8(x)
        x = self.g_s9(x)
        x = self.g_s10(x, w)

        x = self.g_s14(x)
        return x

    def forward(self, x, qmap):
        y = self.g_a(x, qmap) # 3, 3, 3
        z = self.h_a(y, qmap) # 2, 2, 2
        # print(y.shape, z.shape)
        x_hat = self.g_s(y, z)

        return {
            "x_hat": x_hat,
        }

    def decode(self, y_hat, z_hat):
        x_hat = self.g_s(y_hat, z_hat)
        return x_hat

    def encode_decode_noquantize(self, x, qmap):
        enc_start = time.time()
        y = self.g_a(x, qmap) # whole: [1, 16, 16, 16, 16]; [B, 16, 3, 3, 3]
        z = self.h_a(y, qmap) # whole: [1, 16, 8, 8, 8]; [B, 16, 2, 2, 2]
        enc_time = time.time() - enc_start
        dec_start = time.time()
        x_hat = self.g_s(y, z)
        dec_time = time.time() - dec_start
        return {"x_hat": x_hat, "y":y, "z":z}, enc_time, dec_time


# class MySpatiallyAdaptiveCompression_withoutQ(ScaleHyperprior):
class MySpatiallyAdaptiveCompression_withoutQ(nn.Module):
    def __init__(self, N=16, M=16, sft_ks=3, prior_nc=16, inmap=1, ind=1):
        # super().__init__(N, M, **kwargs)
        super(MySpatiallyAdaptiveCompression_withoutQ, self).__init__()
        ### condition networks ###
        # g_a,c
        self.inmap = inmap
        self.ind = ind
        self.qmap_feature_g1 = nn.Sequential(
            conv(inmap, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_g2 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g3 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g4 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
       
        # h_a,c
        self.qmap_feature_h1 = nn.Sequential(
            conv(M + 1, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            # conv(prior_nc * 4, prior_nc * 2, 3, 1),
            # nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_h2 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        

        # f_c
        self.qmap_feature_gs0 = nn.Sequential(
            # nn.ConvTranspose3d(N, N//2, kernel_size=2, stride=1, padding=0), # stride=1 for 3->4, stride=2 for x2;
            nn.ConvTranspose3d(N, N//2, kernel_size=3, stride=2, padding=1),# 2*in-1
            nn.LeakyReLU(0.1, True),
            # UpConv3d(N//2, N//4),
            # nn.LeakyReLU(0.1, True),
            conv(N//2, N//4, 3, 1)
        )

        # g_s,c
        self.qmap_feature_gs1 = nn.Sequential(
            conv(M + N // 4, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_gs2 = nn.Sequential(
            UpConv3d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_gs3 = nn.Sequential(
            UpConv3d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_gs4 = nn.Sequential(
            UpConv3d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        

        ### compression networks ###
        # g_a, encoder
        # self.g_a = None
        self.g_a0 = Conv3d(self.ind, N//4, kernel_size=5, stride=1)
        self.g_a1 = GDN1(N//4)
        self.g_a2 = SFT(N // 4, prior_nc, ks=sft_ks) # output C: N // 4

        self.g_a3 = Conv3d(N//4, N//2)
        self.g_a4 = GDN1(N//2)
        self.g_a5 = SFT(N // 2, prior_nc, ks=sft_ks)

        self.g_a6 = Conv3d(N//2, N)
        self.g_a7 = GDN1(N)
        self.g_a8 = SFT(N, prior_nc, ks=sft_ks)

        self.g_a12 = Conv3d(N, M)
        self.g_a13 = SFTResblk(M, prior_nc, ks=sft_ks)
        self.g_a14 = SFTResblk(M, prior_nc, ks=sft_ks)

        # h_a, hyper encoder
        # self.h_a = None
        self.h_a0 = Conv3d(M, N, kernel_size=3, stride=1)
        self.h_a1 = SFT(N, prior_nc, ks=sft_ks)
        self.h_a2 = nn.LeakyReLU(inplace=True)
        
        self.h_a3 = Conv3d(N, N)
        self.h_a4 = SFTResblk(N, prior_nc, ks=sft_ks)
        self.h_a5 = SFTResblk(N, prior_nc, ks=sft_ks)

        # g_s, decoder
        # self.g_s = None
        self.g_s0 = SFTResblk(M, prior_nc, ks=sft_ks)
        self.g_s1 = SFTResblk(M, prior_nc, ks=sft_ks)

        self.g_s2 = UpConv3d(M, N)
        self.g_s3 = GDN1(N, inverse=True)
        self.g_s4 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s5 = UpConv3d(N, N)
        self.g_s6 = GDN1(N, inverse=True)
        self.g_s7 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s8 = UpConv3d(N, N // 2)
        self.g_s9 = GDN1(N // 2, inverse=True)
        self.g_s10 = SFT(N // 2, prior_nc, ks=sft_ks)

        # self.g_s11 = UpConv3d(N // 2, N // 4)
        # self.g_s12 = GDN1(N // 4, inverse=True)
        # self.g_s13 = SFT(N // 4, prior_nc, ks=sft_ks)

        self.g_s14 = Conv3d(N // 2, self.ind, kernel_size=5, stride=1)

        # h_s, hyper decoder
        self.h_s = nn.Sequential(
            # nn.Upsample(size=3, mode='trilinear', align_corners=False),
            nn.ConvTranspose3d(N, M, kernel_size=2, stride=1, padding=0), # make (2,2,2) to (3,3,3)
            # nn.ConvTranspose3d(N, M, kernel_size=2, stride=2, padding=0),# suppose to X2;
            # nn.ConvTranspose3d(N, M, kernel_size=3, stride=2, padding=1),# 2*in-1
            nn.LeakyReLU(inplace=True),
            conv(M, M * 2, stride=1, kernel_size=3), # padding=kernel_size // 2, M * 2: mu and sigma 
        )

    def g_a(self, x, qmap):
        qmap = self.qmap_feature_g1(qmap)
        x = self.g_a0(x)
        x = self.g_a1(x)
        x = self.g_a2(x, qmap)

        qmap = self.qmap_feature_g2(qmap)
        x = self.g_a3(x)
        x = self.g_a4(x)
        x = self.g_a5(x, qmap)

        qmap = self.qmap_feature_g3(qmap)
        x = self.g_a6(x)
        x = self.g_a7(x)
        x = self.g_a8(x, qmap)

        qmap = self.qmap_feature_g4(qmap)
        x = self.g_a12(x)
        x = self.g_a13(x, qmap)
        x = self.g_a14(x, qmap)
        return x

    def h_a(self, x, qmap):
        # pooling qmap into [3,3,3]
        qmap = F.adaptive_avg_pool3d(qmap, x.size()[2:])
        qmap = self.qmap_feature_h1(torch.cat([qmap, x], dim=1))
        x = self.h_a0(x)
        x = self.h_a1(x, qmap)
        x = self.h_a2(x)

        qmap = self.qmap_feature_h2(qmap)
        x = self.h_a3(x)
        x = self.h_a4(x, qmap)
        x = self.h_a5(x, qmap) #16, 2, 2, 2
        return x

    def g_s(self, x, z):
        w = self.qmap_feature_gs0(z)
        w = self.qmap_feature_gs1(torch.cat([w, x], dim=1))
        x = self.g_s0(x, w)
        x = self.g_s1(x, w)

        w = self.qmap_feature_gs2(w)
        x = self.g_s2(x)
        x = self.g_s3(x)
        x = self.g_s4(x, w)

        w = self.qmap_feature_gs3(w)
        x = self.g_s5(x)
        x = self.g_s6(x)
        x = self.g_s7(x, w)

        w = self.qmap_feature_gs4(w)
        x = self.g_s8(x)
        x = self.g_s9(x)
        x = self.g_s10(x, w)
        
        x = self.g_s14(x)
        return x

    def forward(self, x, qmap):
        y = self.g_a(x, qmap) # 3, 3, 3
        z = self.h_a(y, qmap) # 2, 2, 2
        
        # z_hat, z_likelihoods = self.entropy_bottleneck(z)
        # gaussian_params = self.h_s(z_hat)
        # scales_hat, means_hat = gaussian_params.chunk(chunks=2, dim=1) # split a tensor into 2 chunks
        # y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y, z)
        return {
            "x_hat": x_hat
        }
        
    def encode_decode_noquantize(self, x, qmap):
        enc_start = time.time()
        y = self.g_a(x, qmap) # whole: [1, 16, 16, 16, 16]; [B, 16, 3, 3, 3]
        z = self.h_a(y, qmap) # whole: [1, 16, 8, 8, 8]; [B, 16, 2, 2, 2]
        enc_time = time.time() - enc_start
        dec_start = time.time()
        x_hat = self.g_s(y, z)
        dec_time = time.time() - dec_start
        return {"x_hat": x_hat, "y":y, "z":z}, enc_time, dec_time


class MySpatiallyAdaptiveCompression_no_UpsampledCond(ScaleHyperprior):
    def __init__(self, N=16, M=16, sft_ks=3, prior_nc=16, inmap=2, ind=1, **kwargs):
        super().__init__(N, M, **kwargs)
        ### condition networks ###
        # g_a,c
        self.inmap = inmap
        self.ind = ind
        self.qmap_feature_g1 = nn.Sequential(
            conv(inmap, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_g2 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g3 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g4 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
 
        # h_a,c
        self.qmap_feature_h1 = nn.Sequential(
            conv(M + 1, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            # conv(prior_nc * 4, prior_nc * 2, 3, 1),
            # nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_h2 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        
        # f_c
        self.qmap_feature_gs0 = nn.Sequential(
            nn.ConvTranspose3d(N, N//2, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(0.1, True),
            # UpConv3d(N//2, N//4),
            # nn.LeakyReLU(0.1, True),
            conv(N//2, N//4, 3, 1)
        )

        ### compression networks ###
        # g_a, encoder
        self.g_a = None
        self.g_a0 = Conv3d(self.ind, N//4, kernel_size=5, stride=1)
        self.g_a1 = GDN1(N//4)
        self.g_a2 = SFT(N // 4, prior_nc, ks=sft_ks) # output C: N // 4

        self.g_a3 = Conv3d(N//4, N//2)
        self.g_a4 = GDN1(N//2)
        self.g_a5 = SFT(N // 2, prior_nc, ks=sft_ks)

        self.g_a6 = Conv3d(N//2, N)
        self.g_a7 = GDN1(N)
        self.g_a8 = SFT(N, prior_nc, ks=sft_ks)

        self.g_a12 = Conv3d(N, M)
        self.g_a13 = SFTResblk(M, prior_nc, ks=sft_ks)
        self.g_a14 = SFTResblk(M, prior_nc, ks=sft_ks)

        # h_a, hyper encoder
        self.h_a = None
        self.h_a0 = Conv3d(M, N, kernel_size=3, stride=1)
        self.h_a1 = SFT(N, prior_nc, ks=sft_ks)
        self.h_a2 = nn.LeakyReLU(inplace=True)

        self.h_a3 = Conv3d(N, N)
        self.h_a4 = SFTResblk(N, prior_nc, ks=sft_ks)
        self.h_a5 = SFTResblk(N, prior_nc, ks=sft_ks)

        # g_s, decoder
        self.g_s = None
        self.g_s0 = SFTResblk(M, prior_nc, ks=sft_ks)
        self.g_s1 = SFTResblk(M, prior_nc, ks=sft_ks)

        self.g_s2 = UpConv3d(M, N)
        self.g_s3 = GDN1(N, inverse=True)
        self.g_s4 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s5 = UpConv3d(N, N)
        self.g_s6 = GDN1(N, inverse=True)
        self.g_s7 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s8 = UpConv3d(N, N // 2)
        self.g_s9 = GDN1(N // 2, inverse=True)
        self.g_s10 = SFT(N // 2, prior_nc, ks=sft_ks)

        self.g_s14 = Conv3d(N // 2, self.ind, kernel_size=5, stride=1)

        # h_s, hyper decoder
        self.h_s = nn.Sequential(
            # UpConv3d(N, M),
            # nn.Upsample(size=3, mode='trilinear', align_corners=False),
            nn.ConvTranspose3d(N, M, kernel_size=2, stride=1, padding=0), # make (2,2,2) to (3,3,3)
            # nn.ConvTranspose3d(N, M, kernel_size=2, stride=2, padding=0),# suppose to X2;
            nn.LeakyReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            # conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
            conv(M, M * 2, stride=1, kernel_size=3), # padding=kernel_size // 2, M * 2: mu and sigma 
        )

    def g_a(self, x, qmap):
        qmap = self.qmap_feature_g1(qmap)
        x = self.g_a0(x)
        x = self.g_a1(x)
        x = self.g_a2(x, qmap)

        qmap = self.qmap_feature_g2(qmap)
        x = self.g_a3(x)
        x = self.g_a4(x)
        x = self.g_a5(x, qmap)

        qmap = self.qmap_feature_g3(qmap)
        x = self.g_a6(x)
        x = self.g_a7(x)
        x = self.g_a8(x, qmap)

        qmap = self.qmap_feature_g4(qmap)
        x = self.g_a12(x)
        x = self.g_a13(x, qmap)
        x = self.g_a14(x, qmap)
        return x

    def h_a(self, x, qmap):
        # pooling qmap into [3,3,3]
        qmap = F.adaptive_avg_pool3d(qmap, x.size()[2:])
        qmap = self.qmap_feature_h1(torch.cat([qmap, x], dim=1))
        x = self.h_a0(x)
        x = self.h_a1(x, qmap)
        x = self.h_a2(x)

        qmap = self.qmap_feature_h2(qmap)
        x = self.h_a3(x)
        x = self.h_a4(x, qmap)
        x = self.h_a5(x, qmap) #16, 2, 2, 2
        return x

    def g_s(self, x):
        x = self.g_s2(x)
        x = self.g_s3(x)
        
        x = self.g_s5(x)
        x = self.g_s6(x)
        
        x = self.g_s8(x)
        x = self.g_s9(x)
        x = self.g_s14(x)
        return x

    def forward(self, x, qmap):
        y = self.g_a(x, qmap) # 3, 3, 3
        z = self.h_a(y, qmap) # 2, 2, 2

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(chunks=2, dim=1) # split a tensor into 2 chunks
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, qmap):
        y = self.g_a(x, qmap) # whole: [1, 16, 16, 16, 16]; [B, 16, 3, 3, 3]
        z = self.h_a(y, qmap) # whole: [1, 16, 8, 8, 8]; [B, 16, 2, 2, 2] 
        
        z_strings, _ = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-3:]) # [B, 16, 2, 2, 2]
        # z_strings, _ = self.entropy_bottleneck.torchac_compress(z)
        # z_hat = self.entropy_bottleneck.torchac_decompress(z_strings, z.size()[-3:])
        gaussian_params = self.h_s(z_hat) # [B, 16, 2, 2, 2] --> [B, 16, 3, 3, 3]
        scales_hat, means_hat = gaussian_params.chunk(chunks=2, dim=1) # [B, 16, 3, 3, 3]
        # print(scales_hat.shape, means_hat.shape)
        indexes = self.gaussian_conditional.build_indexes(scales_hat) #  [B, 16, 3, 3, 3]
        y_strings, _ = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-3:], "z_hat":z_hat}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        # z_hat = self.entropy_bottleneck.torchac_decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat)#.clamp_(0, 1)
        return {"x_hat": x_hat, "y_hat": y_hat}

    def decode(self, y_hat):
        x_hat = self.g_s(y_hat)
        return x_hat

    def encode_decode_noquantize(self, x, qmap):
        enc_start = time.time()
        y = self.g_a(x, qmap) # whole: [1, 16, 16, 16, 16]; [B, 16, 3, 3, 3]
        # z = self.h_a(y, qmap) # whole: [1, 16, 8, 8, 8]; [B, 16, 2, 2, 2]
        enc_time = time.time() - enc_start
        dec_start = time.time()
        x_hat = self.g_s(y)
        dec_time = time.time() - dec_start
        return {"x_hat": x_hat}, enc_time, dec_time


"""
single AE without hyper-prior
"""
class MySpatiallyAdaptiveCompressionSingleAE(ScaleHyperprior):
    def __init__(self, N=16, M=16, sft_ks=3, prior_nc=16, **kwargs):
        super().__init__(N, M, **kwargs)
        ### condition networks ###
        # g_a,c
        self.qmap_feature_g1 = nn.Sequential(
            conv(2, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_g2 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g3 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g4 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        
        # g_s,c
        self.qmap_feature_gs1 = nn.Sequential(
            conv(N, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_gs2 = nn.Sequential(
            UpConv3d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_gs3 = nn.Sequential(
            UpConv3d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_gs4 = nn.Sequential(
            UpConv3d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        
        ### compression networks ###
        # g_a, encoder
        self.g_a = None
        self.g_a0 = Conv3d(1, N//4, kernel_size=5, stride=1)
        self.g_a1 = GDN1(N//4)
        self.g_a2 = SFT(N // 4, prior_nc, ks=sft_ks) # output C: N // 4

        self.g_a3 = Conv3d(N//4, N//2)
        self.g_a4 = GDN1(N//2)
        self.g_a5 = SFT(N // 2, prior_nc, ks=sft_ks)

        self.g_a6 = Conv3d(N//2, N)
        self.g_a7 = GDN1(N)
        self.g_a8 = SFT(N, prior_nc, ks=sft_ks)

        self.g_a12 = Conv3d(N, M)
        self.g_a13 = SFTResblk(M, prior_nc, ks=sft_ks)
        self.g_a14 = SFTResblk(M, prior_nc, ks=sft_ks)

        # g_s, decoder
        self.g_s = None
        self.g_s0 = SFTResblk(M, prior_nc, ks=sft_ks)
        self.g_s1 = SFTResblk(M, prior_nc, ks=sft_ks)

        self.g_s2 = UpConv3d(M, N)
        self.g_s3 = GDN1(N, inverse=True)
        self.g_s4 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s5 = UpConv3d(N, N)
        self.g_s6 = GDN1(N, inverse=True)
        self.g_s7 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s8 = UpConv3d(N, N // 2)
        self.g_s9 = GDN1(N // 2, inverse=True)
        self.g_s10 = SFT(N // 2, prior_nc, ks=sft_ks)
        
        self.g_s14 = Conv3d(N // 2, 1, kernel_size=5, stride=1)

    
    def g_a(self, x, qmap):
        # import time
        # starttt = time.time()
        qmap = self.qmap_feature_g1(torch.cat([qmap, x], dim=1))
        x = self.g_a0(x)
        x = self.g_a1(x)
        x = self.g_a2(x, qmap)

        qmap = self.qmap_feature_g2(qmap)
        x = self.g_a3(x)
        x = self.g_a4(x)
        x = self.g_a5(x, qmap)

        qmap = self.qmap_feature_g3(qmap)
        x = self.g_a6(x)
        x = self.g_a7(x)
        x = self.g_a8(x, qmap)

        qmap = self.qmap_feature_g4(qmap)
        x = self.g_a12(x)
        x = self.g_a13(x, qmap)
        x = self.g_a14(x, qmap)
        return x

    def g_s(self, x):
        w = self.qmap_feature_gs1(x)
        x = self.g_s0(x, w)
        x = self.g_s1(x, w)

        w = self.qmap_feature_gs2(w)
        x = self.g_s2(x)
        x = self.g_s3(x)
        x = self.g_s4(x, w)

        w = self.qmap_feature_gs3(w)
        x = self.g_s5(x)
        x = self.g_s6(x)
        x = self.g_s7(x, w)

        w = self.qmap_feature_gs4(w)
        x = self.g_s8(x)
        x = self.g_s9(x)
        x = self.g_s10(x, w)

        x = self.g_s14(x)
        return x

    def forward(self, x, qmap):
        y = self.g_a(x, qmap) 
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods},
        }

    def compress(self, x, qmap):
        y = self.g_a(x, qmap) # whole: [1, 16, 16, 16, 16]; [B, 16, 3, 3, 3]
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings]}

    def decompress(self, string, shape):
        y_hat = self.entropy_bottleneck.decompress(string[0], shape)
        x_hat = self.g_s(y_hat)#.clamp_(0, 1)
        return {"x_hat": x_hat, "y_hat": y_hat}

    def decode(self, y_hat):
        x_hat = self.g_s(y_hat)
        return x_hat


class LatentPredictMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.linear1 = nn.Linear(in_features+4, hidden_features, bias=bias)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.linear2 = nn.Linear(hidden_features, out_features, bias=bias)
        
        # time encoding
        # self.P = np.zeros((128, in_features))
        # X = np.arange(128).reshape(-1, 1) / np.power(
        #     10000, np.arange(0, in_features, 2) / in_features)
        # self.P[:, 0::2] = np.sin(X)
        # self.P[:, 1::2] = np.cos(X)
        # self.P = torch.from_numpy(self.P.astype(np.float32))

        # time concatenate
        self.P = np.zeros((21, 4)) # -90, 90
        X = np.arange(-3.14, 3.14+3.14/10, 3.14/10).reshape(1, -1) 
        self.P[:,0] = np.sin(X)
        self.P[:,1] = np.cos(X)
        self.P[:,2] = np.sin(2*X)
        self.P[:,3] = np.cos(2*X)
        self.P = torch.from_numpy(self.P.astype(np.float32))
        
    def forward(self, start_lat, end_lat, dt):
        """
        :input: Tensor[batch_size, 1, latent_dim] input_latent
        :dt: Tensor[batch_size, 1] delta time 
        :return Tensor[batch_size, 1, latent_dim], predicted residual
        """
        # print(input.shape, dt.shape) # [128, 1, 216], [128]
        start_lat = start_lat.reshape(start_lat.shape[0], -1)
        end_lat = end_lat.reshape(end_lat.shape[0], -1)
        # pdb.set_trace()
        # input_pos = input + self.P[torch.tensor(dt)].to(input.device) #.reshape(input.shape[0], -1) #.as_in_ctx(input.ctx) # position_encoding
        # forward and backward prediction
        forward_pos = torch.cat((start_lat, self.P[torch.tensor(dt+10)].to(start_lat.device)), 1) # forward
        cur_pos = torch.cat((start_lat, self.P[torch.tensor(10).repeat(dt.shape)].to(start_lat.device)), 1)  # current
        backward_pos = torch.cat((end_lat, self.P[torch.tensor(10-dt)].to(end_lat.device)), 1) # backward
        
        out_forward = start_lat + self.linear2(self.leakyrelu(self.linear1(forward_pos)))   # forward
        out_backward = end_lat + self.linear2(self.leakyrelu(self.linear1(backward_pos)))   # backward
        out_cur = start_lat + self.linear2(self.leakyrelu(self.linear1(cur_pos))) 
        return out_forward, out_backward, out_cur
  
