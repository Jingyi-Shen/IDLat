import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

import argparse
import sys
import os

import torch
import numpy as np


from models.models import *
from dataset import get_testdataloader
from utils import load_checkpoint, AverageMeter, get_config, _encode, _decode, _encode_single, _decode_single
from utils import read_gzip_data_noland, read_vortex_data, read_combustion_data, read_nyx_data, read_tornado_data, cal_PSNR_wMSE_wMAE
from utils import pool3d
from losses.losses import Metrics, PixelwiseRateDistortionLoss

from train import get_metric, quality2lambda
from dataset import denormalize_zscore, denormalize_max_min
from dataset import min_max_pressure, mean_std_pressure, min_max_vortex, min_max_combustion_vort, min_max_Tornado_u, min_max_Tornado_v, min_max_Tornado_w, min_max_nyx

import struct 
# import netCDF4 as nc

import pdb
import time

import matplotlib.pyplot as plt

def parse_args(argv):
    parser = argparse.ArgumentParser(description='Pixelwise Variable Rate Compression Evaluation')
    parser.add_argument('--snapshot', help='snapshot path', type=str, required=True)
    parser.add_argument('--output_dir', help='output path', type=str, default='./results/tmp')
    parser.add_argument('--tqdm', help='use tqdm', action='store_true', default=False)
    parser.add_argument('--config', help='config file path', type=str, default='./configs/config_vortex.yaml')
    parser.add_argument('--map_name', help='qmap name, uniform or TF-based or ...', type=str)
    parser.add_argument('--map_value', help='importance map value (uniform case)', default=1., type=float)
    
    args = parser.parse_args(argv)
    
    assert args.snapshot.startswith('./')
    dir_path = '/'.join(args.snapshot.split('/')[:-2])
    # args.config = os.path.join(dir_path, 'config.yaml')
    # args.config = './configs/config_vortex.yaml'
    # args.config = './configs/config_combustion.yaml'
    # args.config = './configs/config_isabel.yaml'
    # args.config = './configs/config_nyx.yaml'
    # args.config = './configs/config_tornado.yaml'
    return args


def test(test_dataloaders, model, criterion, metric, output_dir, dataname='vortex', ts=0, save_latent=False, save_data=False, blocksize=24, padding=4, map_path=None, map_name='uni_1.0'):
    device = next(model.parameters()).device
    loss_all_avg = AverageMeter()
    enc_time_all_avg = AverageMeter()
    dec_time_all_avg = AverageMeter()

    act_size = blocksize - 2 * padding

    with torch.no_grad():
        for i, test_dataloader in enumerate(test_dataloaders):
            bpp_real_avg = AverageMeter()
            psnr_avg = AverageMeter()
            enc_times = AverageMeter()
            dec_times = AverageMeter()

            if dataname == 'isabel':
                result = np.zeros((96, 512, 512))
            elif dataname == 'vortex':
                result = np.zeros((128, 128, 128))
            elif dataname == 'combustion':
                result = np.zeros((128, 720, 480))
            elif dataname == 'nyx' or dataname == 'nyx_ensemble':
                result = np.zeros((256, 256, 256))
            elif dataname == 'tornado':
                result = np.zeros((3, 96, 96, 96))
                
            if save_latent:
                latents_y_hat = []
                latents_z_hat = []

            for x, qmap, z_ind, y_ind, x_ind in test_dataloader:
                x = x.to(device)
                qmap = qmap.to(device)
                # lmbdamap = quality2lambda(qmap)
                
                enc_out, bpp_real, block_byte, enc_time = _encode(model, x, '/tmp/comp', qmap, coder='ans', blocksize=blocksize-2*padding, verbose=False)
                dec_out, dec_time = _decode(model, '/tmp/comp', n_strings=2, shape=enc_out['z_hat'].shape[-3:], coder='ans', verbose=False)
                
                # single y, 
                # enc_out, bpp_real, block_byte, enc_time = _encode_single(model, x, '/tmp/comp', qmap, coder='ans', blocksize=blocksize-2*padding, verbose=False)
                # dec_out, dec_time = _decode_single(model, '/tmp/comp', n_strings=1, shape=enc_out['shape'], coder='ans', verbose=False)
                
                # without quantization
                # enc_out = None
                # bpp_real = 0
                # dec_out, enc_time, dec_time = model.encode_decode_noquantize(x, qmap)

                if save_latent:
                    latents_z_hat.extend(enc_out['z_hat'].detach().cpu().numpy())
                    latents_y_hat.extend(dec_out['y_hat'].detach().cpu().numpy())
                    
                    # latents_z_hat.extend(enc_out['z_symbols'].detach().cpu().numpy())
                    # latents_y_hat.extend(enc_out['y_symbols'].detach().cpu().numpy())

                # out_criterion = criterion(out_net, x, lmbdamap)
                # bpp, psnr = metric(out_net, x)
                psnr = metric.PSNR(dec_out['x_hat'], x)
                
                psnr_avg.update(psnr.item())
                bpp_real_avg.update(bpp_real)
                enc_times.update(enc_time)
                dec_times.update(dec_time)
                
                dec_out = dec_out['x_hat'].detach().cpu().numpy()
                for ii in range(len(z_ind)):
                    if len(result.shape) == 3:
                        if padding > 0:
                            result[int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += dec_out[ii].reshape(blocksize, blocksize, blocksize)[padding:-padding, padding:-padding, padding:-padding]
                            # result_qmap[int(z_ind[ii])*16:int(z_ind[ii])*16+16, int(y_ind[ii])*16:int(y_ind[ii])*16+16, int(x_ind[ii])*16:int(x_ind[ii])*16+16] += lmbdamap[ii].reshape(24, 24, 24)[4:-4, 4:-4, 4:-4]
                        else:
                            result[int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += dec_out[ii].reshape(blocksize, blocksize, blocksize)
                    else:
                        if padding > 0:
                            result[:, int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += dec_out[ii].reshape(-1, blocksize, blocksize, blocksize)[:, padding:-padding, padding:-padding, padding:-padding]
                            # result_qmap[int(z_ind[ii])*16:int(z_ind[ii])*16+16, int(y_ind[ii])*16:int(y_ind[ii])*16+16, int(x_ind[ii])*16:int(x_ind[ii])*16+16] += lmbdamap[ii].reshape(24, 24, 24)[4:-4, 4:-4, 4:-4]
                        else:
                            result[:, int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += dec_out[ii].reshape(-1, blocksize, blocksize, blocksize)
                        
                        
            del x, qmap, z_ind, y_ind, x_ind, enc_out, dec_out
            
            print(
                f'[ Test{ts:03} ]'
                # f' Total: {loss_avg.avg:.4f} |'
                f' Real BPV: {bpp_real_avg.avg:.4f} |'
                # f' BPP: {bpp_avg.avg:.4f} |'
                f' PSNR: {psnr_avg.avg:.4f} |'
                # f' MS-SSIM: {ms_ssim_avg.avg:.4f} |'
                # f' Aux: {aux_loss_avg.avg:.4f} |'
                f' Enc Time: {enc_times.sum:.4f}s |'
                f' Dec Time: {dec_times.sum:.4f}s'
            )

            if dataname == 'isabel':
                val_path = '/fs/project/PAS0027/Isabel_data_pressure/Pf%02d.bin.gz' % (ts)
                ori_data, _ = read_gzip_data_noland(val_path, padding=0)
                min_max = min_max_pressure
                # result = denormalize_zscore(result, mean_std_pressure).reshape(-1)
                # PSNR = cal_PSNR_wMSE_wMAE(ori_data, result, min_max_pressure['max']-min_max_pressure['min'], map_path)
            elif dataname == 'vortex':
                val_path = '/fs/project/PAS0027/vortex_data/vortex/vorts%02d.data' % (ts)
                ori_data, _ = read_vortex_data(val_path, padding=0)
                min_max = min_max_vortex
            elif dataname == 'combustion':
                val_path = '/users/PAS0027/shen1250/Project/SciVis_Autoencoder/data/combustion/combustion/jet_%04d/jet_vort_%04d.dat' % (ts, ts)
                ori_data, _ = read_combustion_data(val_path, padding=0)
                min_max = min_max_combustion_vort
            elif dataname == 'nyx':
                val_path = '/fs/project/PAS0027/nyx/256/256CombineFiles/raw/%d.bin' % (ts)
                ori_data, _ = read_nyx_data(val_path, padding=0)
                min_max = min_max_nyx
            elif dataname == 'tornado':
                val_path = '/users/PAS0027/shen1250/Project/SciVis_Autoencoder/data/tornado/tornado_%02d.nc' % (ts)
                ori_data, _ = read_tornado_data(val_path, padding=0)
                
            elif dataname == 'nyx_ensemble':
                data_filepath_from_map = '/fs/project/PAS0027/nyx/256/output/' + map_name + '/Raw_plt256_00200/density.bin'
                # 'map_0000_0.14903_0.02182_0.83355_thresh10_5.bin' into '0000_0.14903_0.02182_0.83355'
                ori_data, _ = read_nyx_data(data_filepath_from_map, padding=0)
                ori_data = np.log10(ori_data)
                min_max = min_max_nyx
                
            if len(result.shape) == 3:
                ori_data = ori_data.reshape(-1)
                size = result.shape
                result = denormalize_max_min(result, min_max).reshape(-1)
                PSNR, MSE, MAE = cal_PSNR_wMSE_wMAE(ori_data, result, min_max['max']-min_max['min'], map_path, size=size)
                print('PSNR {psnr:.5f}, MAE {mae:.5f}, MSE {mse:.5f}'.format(psnr=PSNR, mae=MAE, mse=MSE)) 
            
            elif len(result.shape) == 4:
                ori_data = ori_data.reshape(3, -1)
                result[0] = denormalize_max_min(result[0], min_max_Tornado_u)
                result[1] = denormalize_max_min(result[1], min_max_Tornado_v)
                result[2] = denormalize_max_min(result[2], min_max_Tornado_w)
                size = result.shape[1:]
                
                PSNR_u, MSE_u, MAE_u = cal_PSNR_wMSE_wMAE(ori_data[0], result[0].reshape(-1), min_max_Tornado_u['max']-min_max_Tornado_u['min'], map_path, size=size)
                print('u PSNR {psnr:.5f}, MAE {mae:.5f}, MSE {mse:.5f}'.format(psnr=PSNR_u, mae=MAE_u, mse=MSE_u)) 
                PSNR_v, MSE_v, MAE_v = cal_PSNR_wMSE_wMAE(ori_data[1], result[1].reshape(-1), min_max_Tornado_v['max']-min_max_Tornado_v['min'], map_path, size=size)
                print('v PSNR {psnr:.5f}, MAE {mae:.5f}, MSE {mse:.5f}'.format(psnr=PSNR_v, mae=MAE_v, mse=MSE_v)) 
                PSNR_w, MSE_w, MAE_w = cal_PSNR_wMSE_wMAE(ori_data[2], result[2].reshape(-1), min_max_Tornado_w['max']-min_max_Tornado_w['min'], map_path, size=size)
                print('w PSNR {psnr:.5f}, MAE {mae:.5f}, MSE {mse:.5f}'.format(psnr=PSNR_w, mae=MAE_w, mse=MSE_w)) 
                
                PSNR = (PSNR_u + PSNR_v + PSNR_w ) / 3.
                MSE = (MSE_u + MSE_v + MSE_w ) / 3.
                MAE = (MAE_u + MAE_v + MAE_w ) / 3.
                result = result.reshape(3,-1).T.reshape(-1)
            
            if save_latent:
                print('save latents!')
                print(np.array(latents_z_hat).shape, np.array(latents_y_hat).shape)
                if dataname == 'nyx_ensemble':
                    np.save(os.path.join(output_dir+'latents/', dataname+f'_y_hat_{map_name}_bpp{bpp_real_avg.avg:.4f}_psnr{PSNR:.4f}.npy'), latents_y_hat)
                    np.save(os.path.join(output_dir+'latents/', dataname+f'_z_hat_{map_name}_bpp{bpp_real_avg.avg:.4f}_psnr{PSNR:.4f}.npy'), latents_z_hat)
                
                else:
                    np.save(os.path.join(output_dir+'latents/', dataname[:3]+f'_{ts:03}_y_hat_{map_name}_bpp{bpp_real_avg.avg:.4f}_psnr{PSNR:.4f}.npy'), latents_y_hat)
                    np.save(os.path.join(output_dir+'latents/', dataname[:3]+f'_{ts:03}_z_hat_{map_name}_bpp{bpp_real_avg.avg:.4f}_psnr{PSNR:.4f}.npy'), latents_z_hat)
                del latents_z_hat, latents_y_hat
            
            del ori_data
            if save_data:
                print(np.max(result), np.min(result))
                with open(os.path.join(output_dir, dataname[:3]+f'_{ts:03}_{map_name}_thresh_bpp{bpp_real_avg.avg:.4f}_psnr{PSNR:.4f}.bin'), 'wb') as f:
                    f.write(struct.pack('<%df' % len(result), *result))
            del result
            
            enc_time_all_avg.update(enc_times.sum)
            dec_time_all_avg.update(dec_times.sum)

        print(f'[ Test ] Total mean: {loss_all_avg.avg:.4f} |' 
              f' Enc Time: {enc_time_all_avg.avg:.4f}s |'
              f' Dec Time: {dec_time_all_avg.avg:.4f}s')


def test_withoutQ(test_dataloaders, model, criterion, metric, output_dir, dataname='vortex', ts=0, save_latent=False, save_data=False, blocksize=24, padding=4, map_path=None, map_name='uni_1.0'):
    device = next(model.parameters()).device
    loss_all_avg = AverageMeter()
    enc_time_all_avg = AverageMeter()
    dec_time_all_avg = AverageMeter()

    act_size = blocksize - 2 * padding

    with torch.no_grad():
        for i, test_dataloader in enumerate(test_dataloaders):
            bpp_real_avg = AverageMeter()
            psnr_avg = AverageMeter()
            # ms_ssim_avg = AverageMeter()
            enc_times = AverageMeter()
            dec_times = AverageMeter()

            if dataname == 'isabel':
                result = np.zeros((96, 512, 512))
            elif dataname == 'vortex':
                result = np.zeros((128, 128, 128))
            elif dataname == 'combustion':
                result = np.zeros((128, 720, 480))
            elif dataname == 'nyx' or dataname == 'nyx_ensemble':
                result = np.zeros((256, 256, 256))
            elif dataname == 'tornado':
                result = np.zeros((3, 96, 96, 96))
                
            if save_latent:
                latents_y_hat = []
                latents_z_hat = []

            for x, qmap, z_ind, y_ind, x_ind in test_dataloader:
                x = x.to(device)
                qmap = qmap.to(device)
                
                # out_net, enc_time, dec_time = model.encode_decode_noquantize(x, qmap) # return {"x_hat": x_hat, "y":y, "z":z}, enc_time, dec_time
                out_net, enc_time = model.encoding_yz(x, qmap)
                
                if save_latent:
                    latents_z_hat.extend(out_net['z'].detach().cpu().numpy())
                    latents_y_hat.extend(out_net['y'].detach().cpu().numpy())

                # psnr = metric.PSNR(out_net['x_hat'], x)
                
                # psnr_avg.update(psnr.item())
                enc_times.update(enc_time)
                # dec_times.update(dec_time)
                
                dec_out = out_net['x_hat'].detach().cpu().numpy()
                for ii in range(len(z_ind)):
                    if len(result.shape) == 3:
                        if padding > 0:
                            result[int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += dec_out[ii].reshape(blocksize, blocksize, blocksize)[padding:-padding, padding:-padding, padding:-padding]
                            # result_qmap[int(z_ind[ii])*16:int(z_ind[ii])*16+16, int(y_ind[ii])*16:int(y_ind[ii])*16+16, int(x_ind[ii])*16:int(x_ind[ii])*16+16] += lmbdamap[ii].reshape(24, 24, 24)[4:-4, 4:-4, 4:-4]
                        else:
                            result[int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += dec_out[ii].reshape(blocksize, blocksize, blocksize)
                    else:
                        if padding > 0:
                            result[:, int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += dec_out[ii].reshape(-1, blocksize, blocksize, blocksize)[:, padding:-padding, padding:-padding, padding:-padding]
                            # result_qmap[int(z_ind[ii])*16:int(z_ind[ii])*16+16, int(y_ind[ii])*16:int(y_ind[ii])*16+16, int(x_ind[ii])*16:int(x_ind[ii])*16+16] += lmbdamap[ii].reshape(24, 24, 24)[4:-4, 4:-4, 4:-4]
                        else:
                            result[:, int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += dec_out[ii].reshape(-1, blocksize, blocksize, blocksize)
                         
            del x, qmap, z_ind, y_ind, x_ind, out_net
            
            print(
                f'[ Test{ts:03} ]'
                # f' Total: {loss_avg.avg:.4f} |'
                f' PSNR: {psnr_avg.avg:.4f} |'
                f' Enc Time: {enc_times.sum:.4f}s |'
                f' Dec Time: {dec_times.sum:.4f}s'
            )

            if dataname == 'isabel':
                val_path = '/fs/project/PAS0027/Isabel_data_pressure/Pf%02d.bin.gz' % (ts)
                ori_data, _ = read_gzip_data_noland(val_path, padding=0)
                min_max = min_max_pressure
                # result = denormalize_zscore(result, mean_std_pressure).reshape(-1)
                # PSNR = cal_PSNR_wMSE_wMAE(ori_data, result, min_max_pressure['max']-min_max_pressure['min'], map_path)
            elif dataname == 'vortex':
                val_path = '/fs/project/PAS0027/vortex_data/vortex/vorts%02d.data' % (ts)
                ori_data, _ = read_vortex_data(val_path, padding=0)
                min_max = min_max_vortex
            elif dataname == 'combustion':
                val_path = '/users/PAS0027/shen1250/Project/SciVis_Autoencoder/data/combustion/combustion/jet_%04d/jet_vort_%04d.dat' % (ts, ts)
                ori_data, _ = read_combustion_data(val_path, padding=0)
                min_max = min_max_combustion_vort
            elif dataname == 'nyx':
                val_path = '/fs/project/PAS0027/nyx/256/256CombineFiles/raw/%d.bin' % (ts)
                ori_data, _ = read_nyx_data(val_path, padding=0)
                min_max = min_max_nyx
            elif dataname == 'tornado':
                val_path = '/users/PAS0027/shen1250/Project/SciVis_Autoencoder/data/tornado/tornado_%02d.nc' % (ts)
                ori_data, _ = read_tornado_data(val_path, padding=0)
            
            elif dataname == 'nyx_ensemble':
                data_filepath_from_map = '/fs/project/PAS0027/nyx/256/output/' + map_name + '/Raw_plt256_00200/density.bin'
                # 'map_0000_0.14903_0.02182_0.83355_thresh10_5.bin' into '0000_0.14903_0.02182_0.83355'
                ori_data, _ = read_nyx_data(data_filepath_from_map, padding=0)
                ori_data = np.log10(ori_data)
                min_max = min_max_nyx
                
            if len(result.shape) == 3:
                ori_data = ori_data.reshape(-1)
                size = result.shape
                result = denormalize_max_min(result, min_max).reshape(-1)
                PSNR, MSE, MAE = cal_PSNR_wMSE_wMAE(ori_data, result, min_max['max']-min_max['min'], map_path, size=size)
                print('PSNR {psnr:.5f}, MAE {mae:.5f}, MSE {mse:.5f}'.format(psnr=PSNR, mae=MAE, mse=MSE)) 
            
            elif len(result.shape) == 4:
                ori_data = ori_data.reshape(3, -1)
                result[0] = denormalize_max_min(result[0], min_max_Tornado_u)
                result[1] = denormalize_max_min(result[1], min_max_Tornado_v)
                result[2] = denormalize_max_min(result[2], min_max_Tornado_w)
                size = result.shape[1:]
                
                PSNR_u, MSE_u, MAE_u = cal_PSNR_wMSE_wMAE(ori_data[0], result[0].reshape(-1), min_max_Tornado_u['max']-min_max_Tornado_u['min'], map_path, size=size)
                print('u PSNR {psnr:.5f}, MAE {mae:.5f}, MSE {mse:.5f}'.format(psnr=PSNR_u, mae=MAE_u, mse=MSE_u)) 
                PSNR_v, MSE_v, MAE_v = cal_PSNR_wMSE_wMAE(ori_data[1], result[1].reshape(-1), min_max_Tornado_v['max']-min_max_Tornado_v['min'], map_path, size=size)
                print('v PSNR {psnr:.5f}, MAE {mae:.5f}, MSE {mse:.5f}'.format(psnr=PSNR_v, mae=MAE_v, mse=MSE_v)) 
                PSNR_w, MSE_w, MAE_w = cal_PSNR_wMSE_wMAE(ori_data[2], result[2].reshape(-1), min_max_Tornado_w['max']-min_max_Tornado_w['min'], map_path, size=size)
                print('w PSNR {psnr:.5f}, MAE {mae:.5f}, MSE {mse:.5f}'.format(psnr=PSNR_w, mae=MAE_w, mse=MSE_w)) 
                
                PSNR = (PSNR_u + PSNR_v + PSNR_w ) / 3.
                MSE = (MSE_u + MSE_v + MSE_w ) / 3.
                MAE = (MAE_u + MAE_v + MAE_w ) / 3.
                result = result.reshape(3,-1).T.reshape(-1)
            
            if save_latent:
                print('save latents!')
                print(np.array(latents_z_hat).shape, np.array(latents_y_hat).shape)
                if dataname == 'nyx_ensemble':
                    np.save(os.path.join(output_dir+'latents_raw_noQ/', dataname+f'_y_hat_{map_name}_bpp{bpp_real_avg.avg:.4f}_psnr{PSNR:.4f}_woQ.npy'), latents_y_hat)
                    np.save(os.path.join(output_dir+'latents_raw_noQ/', dataname+f'_z_hat_{map_name}_bpp{bpp_real_avg.avg:.4f}_psnr{PSNR:.4f}_woQ.npy'), latents_z_hat)
                
                else:
                    np.save(os.path.join(output_dir+'latents_raw_noQ/', dataname[:3]+f'_{ts:03}_y_hat_{map_name}_bpp{bpp_real_avg.avg:.4f}_psnr{PSNR:.4f}_woQ.npy'), latents_y_hat)
                    np.save(os.path.join(output_dir+'latents_raw_noQ/', dataname[:3]+f'_{ts:03}_z_hat_{map_name}_bpp{bpp_real_avg.avg:.4f}_psnr{PSNR:.4f}_woQ.npy'), latents_z_hat)
                del latents_z_hat, latents_y_hat
            
            del ori_data
            if save_data:
                print(np.max(result), np.min(result))
                with open(os.path.join(output_dir, dataname[:3]+f'_{ts:03}_bpp{bpp_real_avg.avg:.4f}_psnr{PSNR:.4f}_{map_name}_woQ.bin'), 'wb') as f:
                    f.write(struct.pack('<%df' % len(result), *result))
                # with open(f'./data/combustion/{ts:03}.bin' , 'wb') as f:
                #     f.write(struct.pack('<%df' % len(ori_data), *ori_data))
            del result
            
            enc_time_all_avg.update(enc_times.sum)
            dec_time_all_avg.update(dec_times.sum)

        print(f'[ Test ] Total mean: {loss_all_avg.avg:.4f} |' 
              f' Enc Time: {enc_time_all_avg.avg:.4f}s |'
              f' Dec Time: {dec_time_all_avg.avg:.4f}s')


def main(argv):
    args = parse_args(argv)
    config = get_config(args.config)
    # config['testset'] = args.testset
    print('[config]', args.config)
    
    msg = f'======================= {args.snapshot} ======================='
    print(msg)
    for k, v in config.items():
        if k in {'lr', 'set_lr', 'p', 'testset'}:
            print(f' *{k}: ', v)
        else:
            print(f'  {k}: ', v)
    print('=' * len(msg))
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print('device=', device)
    metric = get_metric(dataname=config['dataname'], block_size=config['blocksize']-2*config['padding'])
    criterion = PixelwiseRateDistortionLoss(config['beta'], block_size=config['blocksize']-2*config['padding'], dataname=config['dataname'])
    
    # model = MySpatiallyAdaptiveCompression_lighter(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'], inmap=config['inmap'], ind=config['ind'])
    model = MySpatiallyAdaptiveCompression(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'], inmap=config['inmap'], ind=config['ind'])
    # model = MySpatiallyAdaptiveCompressionSingleAE(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'])
    # model = MySpatiallyAdaptiveCompression_withoutQ(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'], inmap=config['inmap'], ind=config['ind'])
    # model = MySpatiallyAdaptiveCompression_AE(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'], inmap=config['inmap'], ind=config['ind'])
        
    model = model.to(device)
    itr, model = load_checkpoint(args.snapshot, model, only_net=True)
    model.update() # necessary 
    
    # cdf = model.entropy_bottleneck._quantized_cdf.detach().cpu().numpy()
    # offset = model.entropy_bottleneck._offset.detach().cpu().numpy()
    # cdf_length = model.entropy_bottleneck._cdf_length.detach().cpu().numpy()
    # print(cdf)
    # print(offset)
    # print(cdf_length)
    # print(cdf.shape)
    # for index in range(len(cdf)):
    #     cdf_l = cdf_length[index]
    #     x = [i for i in range(cdf_l)]
    #     plt.plot(x, cdf[index][:cdf_l], label=index)
    # x = [i for i in range(max(cdf_length))]
    # x_name = [min(offset)+i for i in range(max(cdf_length))]
    # plt.xticks(x, x_name)
    # plt.show()
    # exit()
    # for p in model.aux_parameters():
    #     print(p.name, p.data)

    # whole volume testing of all blocks
    test_dataloaders = get_testdataloader(config, uniform_value=args.map_value)
    ts = int(config['test_data_file_path'].split('/')[-1].split('.')[0][-2:])
    test(test_dataloaders, model, criterion, metric, args.output_dir, dataname=config['dataname'], ts=ts, save_latent=True, save_data=False, blocksize=config['blocksize'], padding=config['padding'], map_path=config['data_map_file_path'], map_name=args.map_name)
    # test_withoutQ(test_dataloaders, model, criterion, metric, args.output_dir, dataname=config['dataname'], ts=ts, save_latent=False, save_data=True, blocksize=config['blocksize'], padding=config['padding'], map_path=config['data_map_file_path'], map_name=args.map_name)


# def main_ensemble_nyx(argv):
#     args = parse_args(argv)
#     config = get_config(args.config)
#     # config['testset'] = args.testset
#     print('[config]', args.config)
    
#     eval_dir = '/fs/project/PAS0027/nyx/256/output/'
#     param_dirs = np.array([f for f in sorted(os.listdir(eval_dir)) ])[:800]
#     files = np.array([f+'/Raw_plt256_00200/density.bin' for f in  param_dirs])
    
#     eval_map_dir = './data/nyx_high_density_map_ensemble/'
#     eval_map_files = np.array([f for f in sorted(os.listdir(eval_map_dir)) ])
    
#     for ensemble_idx in range(0, 800):
#         config['dataname'] = 'nyx_ensemble'
#         config['test_data_file_path'] = eval_dir + files[ensemble_idx]
#         config['data_map_file_path'] = eval_map_dir + eval_map_files[ensemble_idx]
        
#         map_name_ =  eval_map_files[ensemble_idx][4:][:-15] 
#         args.map_name = map_name_
        
#         msg = f'======================= {args.snapshot} ======================='
#         print(msg)
#         for k, v in config.items():
#             if k in {'lr', 'set_lr', 'p', 'testset'}:
#                 print(f' *{k}: ', v)
#             else:
#                 print(f'  {k}: ', v)
#         print('=' * len(msg))
#         print()
    
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         # device = 'cpu'
#         print('device=', device)
#         metric = get_metric(dataname='nyx', block_size=config['blocksize']-2*config['padding'])
#         criterion = PixelwiseRateDistortionLoss(config['beta'], block_size=config['blocksize']-2*config['padding'], dataname='nyx')
        
#         model = MySpatiallyAdaptiveCompression(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'], inmap=config['inmap'], ind=config['ind'])
#         # model = MySpatiallyAdaptiveCompression_withoutQ(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'], inmap=config['inmap'], ind=config['ind'])
        
#         model = model.to(device)
        
#         itr, model = load_checkpoint(args.snapshot, model, only_net=True)
#         model.update()
        
#         # whole volume testing of all blocks
#         test_dataloaders = get_testdataloader(config, uniform_value=args.map_value)
#         ts = 0 #int(config['test_data_file_path'].split('/')[-1].split('.')[0][-2:])
#         test(test_dataloaders, model, criterion, metric, args.output_dir, dataname=config['dataname'], ts=ts, save_latent=True, save_data=False, blocksize=config['blocksize'], padding=config['padding'], map_path=config['data_map_file_path'], map_name=map_name_)
#         # test_withoutQ(test_dataloaders, model, criterion, metric, args.output_dir, dataname=config['dataname'], ts=ts, save_latent=True, save_data=False, blocksize=config['blocksize'], padding=config['padding'], map_path=config['data_map_file_path'], map_name=map_name_)

        
# def main_ensemble_vortex(argv):
#     args = parse_args(argv)
#     config = get_config(args.config)
#     # config['testset'] = args.testset
#     print('[config]', args.config)
    
#     file = '/fs/project/PAS0027/vortex_data/vortex/vorts01.data'
    
#     eval_map_dir = './data/iso_bin/vorts01_dense_bin/'
#     # eval_map_dir =  './data/iso_bin/vorts01_dense_bin/'
#     eval_map_files = np.array([f for f in sorted(os.listdir(eval_map_dir)) if f.endswith('.bin') and f.startswith('vorts') ])
#     config['dataname'] = 'vortex'  # map_01_iso4.0.bin
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     # device = 'cpu'
#     print('device=', device)
    
#     metric = get_metric(dataname=config['dataname'], block_size=config['blocksize']-2*config['padding'])
#     criterion = PixelwiseRateDistortionLoss(config['beta'], block_size=config['blocksize']-2*config['padding'], dataname=config['dataname'])
        
#     model = MySpatiallyAdaptiveCompression(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'], inmap=config['inmap'], ind=config['ind'])
#     # model = MySpatiallyAdaptiveCompression_withoutQ(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'], inmap=config['inmap'], ind=config['ind'])
#     # model = MySpatiallyAdaptiveCompression_AE(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'], inmap=config['inmap'], ind=config['ind'])
        
#     model = model.to(device)
        
#     itr, model = load_checkpoint(args.snapshot, model, only_net=True)
#     model.update()
    
#     start = time.time()
    
#     for iso_idx in range(len(eval_map_files)):
#         config['data_map_file_path'] = eval_map_dir + eval_map_files[iso_idx]
#         map_name_ = eval_map_files[iso_idx][:-4] # 'AE_noQ'
#         # map_name_ = eval_map_files[iso_idx][8:-4]
#         args.map_name = map_name_
#         # msg = f'======================= {args.snapshot} ======================='
#         # print(msg)
#         # for k, v in config.items():
#         #     if k in {'lr', 'set_lr', 'p', 'testset'}:
#         #         print(f' *{k}: ', v)
#         #     else:
#         #         print(f'  {k}: ', v)
#         # print('=' * len(msg))
#         # print()
#         print(map_name_)
    
#         # whole volume testing of all blocks
#         test_dataloaders = get_testdataloader(config, uniform_value=args.map_value)
#         ts = 1 #int(config['test_data_file_path'].split('/')[-1].split('.')[0][-2:])
#         # test_withoutQ(test_dataloaders, model, criterion, metric, args.output_dir, dataname=config['dataname'], ts=ts, save_latent=False, save_data=False, blocksize=config['blocksize'], padding=config['padding'], map_path=config['data_map_file_path'], map_name=args.map_name)
#         test(test_dataloaders, model, criterion, metric, args.output_dir, dataname=config['dataname'], ts=ts, save_latent=False, save_data=False, blocksize=config['blocksize'], padding=config['padding'], map_path=config['data_map_file_path'], map_name=args.map_name)
    
#     end = time.time()
#     print('time:', end-start)


if __name__ == '__main__':
    main(sys.argv[1:])
    # main_ensemble_nyx(sys.argv[1:])
    # main_ensemble_vortex(sys.argv[1:])

