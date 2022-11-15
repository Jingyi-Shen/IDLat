import argparse
from ctypes import Structure
import random
import sys
import os
from datetime import datetime
import numpy as np
import struct

import torch
import torch.optim as optim
import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity

from models.models import *
from dataset import get_train_test_dataloader
from dataset import min_max_pressure, mean_std_pressure, min_max_vortex, min_max_combustion_vort, min_max_nyx, min_max_Tornado_u, min_max_Tornado_v, min_max_Tornado_w 
from utils import init, Logger, load_checkpoint, save_checkpoint, AverageMeter, get_duration

from losses.losses import Metrics, PixelwiseRateDistortionLoss, DistortionLoss
from dataset import denormalize_zscore, denormalize_max_min


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Spatially-Adaptive Variable Rate Compression')
    parser.add_argument('--train', action='store_true', help='Is true if is training.')
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--name', help='result dir name', default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), type=str)
    parser.add_argument('--resume', help='snapshot path', type=str)
    parser.add_argument('--seed', help='seed number', default=123, type=int)
    args = parser.parse_args(argv)

    if not args.config:
        if args.resume:
            assert args.resume.startswith('./')
            dir_path = '/'.join(args.resume.split('/')[:-2])
            args.config = os.path.join(dir_path, 'config.yaml')
        else:
            args.config = './configs/config.yaml'

    return args


def quality2lambda(qmap, a=5.5):
    return torch.exp(a * qmap) # 
    # return torch.exp(5.5 * qmap) # tornado
    # return 0.1*torch.exp(5.5 * qmap) # vortex
    # return 0.01*torch.exp(5.5 * qmap)
    # return 1e-3 * torch.exp(7 * qmap)
    # return 1e-3 * torch.exp(4.382 * qmap)

def get_metric(dataname, block_size, Q=False):
    if dataname == 'isabel':
        # metric = Metrics(norm_method='mean_std', mean_std_norm=mean_std_pressure, min_max_norm=min_max_pressure)
        metric = Metrics(norm_method='min_max', min_max_norm=min_max_pressure, block_size=block_size, Q=Q)
    elif dataname == 'vortex':
        metric = Metrics(norm_method='min_max', min_max_norm=min_max_vortex, block_size=block_size, Q=Q)
    elif dataname == 'combustion':
        metric = Metrics(norm_method='min_max', min_max_norm=min_max_combustion_vort, block_size=block_size, Q=Q)
    elif dataname == 'nyx':
        metric = Metrics(norm_method='min_max', min_max_norm=min_max_nyx, block_size=block_size, Q=Q)
    elif dataname == 'tornado':
        metric = Metrics(norm_method='min_max', min_max_norm=min_max_Tornado_u, block_size=block_size, Q=Q)
    return metric


# def test_withoutQ(dataname, logger, test_dataloader, model, criterion, metric, val_path, blocksize=24, padding=4, a=1):
#     model.eval()
#     device = next(model.parameters()).device
#     loss = AverageMeter()
#     bpp_loss = AverageMeter()
#     mse_loss = AverageMeter()

#     act_size = blocksize - 2* padding

#     with torch.no_grad():
#         logger.init()
#         if dataname == 'isabel':
#             result = np.zeros((96, 512, 512))
#         elif dataname == 'vortex':
#             result = np.zeros((128, 128, 128))
#         elif dataname == 'combustion':
#             result = np.zeros((128, 720, 480))
#         elif dataname == 'nyx':
#             result = np.zeros((256, 256, 256))
#         elif dataname == 'tornado':
#             result = np.zeros((3, 96, 96, 96))
            
#         start = datetime.now()
#         for x, qmap, z_ind, y_ind, x_ind in test_dataloader:
#             x = x.to(device)
#             qmap = qmap.to(device)
#             lmbdamap = quality2lambda(qmap, a)
#             # lmbdamap = qmap
#             out_net = model(x, qmap)
            
#             # for name, param in model.state_dict().items():
#             #     print(name, param.size())
#             out_criterion = criterion(out_net, x, lmbdamap)
#             bpp, psnr = metric(out_net, x)
#             logger.update_test(bpp, psnr, out_criterion, model.aux_loss())
#             out = out_net['x_hat'].detach().cpu().numpy()
            
#             for ii in range(len(z_ind)):
#                 if len(result.shape) == 3:
#                     if padding > 0:
#                         result[int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += out[ii].reshape(blocksize, blocksize, blocksize)[padding:-padding, padding:-padding, padding:-padding]
#                     else:
#                         result[int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += out[ii].reshape(blocksize, blocksize, blocksize)
#                 else:
#                     if padding > 0:
#                         result[:,int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += out[ii].reshape(-1, blocksize, blocksize, blocksize)[:, padding:-padding, padding:-padding, padding:-padding]
#                     else:
#                         result[:,int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += out[ii].reshape(-1, blocksize, blocksize, blocksize)
        
#         del x, qmap, z_ind, y_ind, x_ind, out_net, out_criterion, lmbdamap
        
#         h, m, s = get_duration(start, datetime.now())
#         time_str = 'Test Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)  
#         print(time_str)
        
        
#         loss.update(logger.loss.avg)
#         bpp_loss.update(logger.bpp_loss.avg)
#         mse_loss.update(logger.mse_loss.avg)
        
#         if dataname == 'isabel':
#             result = denormalize_max_min(result, min_max_pressure).reshape(-1)
#             # result = denormalize_zscore(result, mean_std_pressure).reshape(-1)
#         elif dataname == 'vortex':
#             result = denormalize_max_min(result, min_max_vortex).reshape(-1)
#         elif dataname == 'combustion':
#             result = denormalize_max_min(result, min_max_combustion_vort).reshape(-1)
#         elif dataname == 'nyx':
#             result = denormalize_max_min(result, min_max_nyx).reshape(-1)
#         elif dataname == 'tornado':
#             result[0] = denormalize_max_min(result[0], min_max_Tornado_u)
#             result[1] = denormalize_max_min(result[1], min_max_Tornado_v)
#             result[2] = denormalize_max_min(result[2], min_max_Tornado_w)
            
#             result = result.reshape(3,-1).T
#             result = result.reshape(-1)
            
#         with open(os.path.join(logger.output_dir, dataname[:3]+f'_bpp{logger.bpp_loss.avg:.4f}_mse{logger.mse_loss.avg:.4f}_loss{logger.loss.avg:.4f}.bin'), 'wb') as f:
#             f.write(struct.pack('<%df' % len(result), *result))
        
#         # del all_data, result
#         # print(
#         #     f' BPP all: {bpp_all:.4f} |'
#         #     f' PSNR all: {psnr_all:.4f} |'
#         #     f' MS SSIM all: {ms_ssim_all:.4f}')
#         print(f'[ Test ] Total mean: {loss.avg:.4f}')
        
#     logger.init()
#     model.train()

#     return loss.avg, bpp_loss.avg, mse_loss.avg
   

# def train_withoutQ(args, config, base_dir, snapshot_dir, output_dir, log_dir):
#     device = torch.device("cuda" if config['cuda'] else "cpu") 
#     # device = 'cpu'
#     print('device', device)
    
#     # criterion = PixelwiseRateDistortionLoss(config['beta'], block_size=config['blocksize']-2*config['padding'], dataname=config['dataname'])
#     criterion = DistortionLoss(config['beta'], block_size=config['blocksize']-2*config['padding'], dataname=config['dataname'])
#     metric = get_metric(dataname=config['dataname'], block_size=config['blocksize']-2*config['padding'], Q=True)
#     train_dataloader, test_dataloader = get_train_test_dataloader(config)
#     logger = Logger(config, base_dir, snapshot_dir, output_dir, log_dir)

#     # model = MySpatiallyAdaptiveCompression_lighter(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'], inmap=config['inmap'], ind=config['ind'])
#     # model = MySpatiallyAdaptiveCompression(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'], inmap=config['inmap'], ind=config['ind'])
#     # model = MySpatiallyAdaptiveCompression_withoutQ(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'], inmap=config['inmap'], ind=config['ind'])
#     # model = MySpatiallyAdaptiveCompressionSingleAE(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'])
#     # model = MySpatiallyAdaptiveCompressionPowerOfTwo(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'])
#     # model = MySpatiallyAdaptiveCompressionPowerOfTwo_onlyQMAP(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'])
#     # model = MySpatiallyAdaptiveCompression_no_UpsampledCond(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'], inmap=config['inmap'], ind=config['ind'])
#     # model = MySpatiallyAdaptiveCompression_AE(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'], inmap=config['inmap'], ind=config['ind'])
#     model = MySpatiallyAdaptiveCompression_AE_IMP(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'], inmap=config['inmap'], ind=config['ind'])
    
#     model = model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=config['lr'])

#     if config['set_lr']:
#         lr_prior = optimizer.param_groups[0]['lr']
#         for g in optimizer.param_groups:
#             g['lr'] = float(config['set_lr'])
#         print(f'[set lr] {lr_prior} -> {optimizer.param_groups[0]["lr"]}')

#     a = config['a']
#     # b = config['b']

#     model.train()
#     loss_best = 1e10
#     while logger.itr < config['max_itr']:
#         # print('logger.itr', logger.itr)
#         for x, qmap in train_dataloader:
#             optimizer.zero_grad()
            
#             x = x.to(device)
#             qmap = qmap.to(device)
#             lmbdamap = quality2lambda(qmap, a)
#             # lmbdamap = qmap

#             out_net = model(x, qmap)

#             out_criterion = criterion(out_net, x, lmbdamap)
#             # import pdb
#             # pdb.set_trace()
            
#             out_criterion['loss'].backward()
#             # for stability
#             if out_criterion['loss'].isnan().any() or out_criterion['loss'].isinf().any() or out_criterion['loss'] > 10000:
#                 continue

#             # if config['clip_max_norm'] > 0:
#             #     torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_max_norm'])
#             optimizer.step()

#         # logging
#         logger.update(out_criterion, aux_loss=out_criterion['loss'])
#         if logger.itr % config['log_itr'] == 0:
#             logger.print()
#             logger.write()
#             logger.init()

#         # test and save model snapshot
#         if logger.itr % config['test_itr'] == 0 or logger.itr % config['snapshot_save_itr'] == 0:
#             # model.update()
#             loss, bpp_loss, mse_loss = test(config['dataname'], logger, test_dataloader, model, criterion, metric, 
#                                             val_path=config['test_data_file_path'], blocksize=config['blocksize'], padding=config['padding'],
#                                             a=a, Q=True)
#             # model.train()
#             if loss < loss_best:
#                 print('Best!')
#                 save_checkpoint(os.path.join(snapshot_dir, 'best.pt'), logger.itr, model, optimizer)
#                 loss_best = loss
#             if logger.itr % config['snapshot_save_itr'] == 0:
#                 save_checkpoint(os.path.join(snapshot_dir, f'{logger.itr:05}_{bpp_loss:.4f}_{mse_loss:.8f}.pt'),
#                                 logger.itr, model, optimizer)

#         # lr scheduling
#         if logger.itr % config['lr_shedule_step'] == 0:
#             lr_prior = optimizer.param_groups[0]['lr']
#             for g in optimizer.param_groups:
#                 g['lr'] *= config['lr_shedule_scale']
#             print(f'[lr scheduling] {lr_prior} -> {optimizer.param_groups[0]["lr"]}')

#     del x, qmap, out_net, lmbdamap


def test(dataname, logger, test_dataloader, model, criterion, metric, val_path, blocksize=24, padding=4, a=1, Q=False):
    model.eval()
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()

    act_size = blocksize - 2* padding

    with torch.no_grad():
        logger.init()
        if dataname == 'isabel':
            result = np.zeros((96, 512, 512))
        elif dataname == 'vortex':
            result = np.zeros((128, 128, 128))
        elif dataname == 'combustion':
            result = np.zeros((128, 720, 480))
        elif dataname == 'nyx':
            result = np.zeros((256, 256, 256))
        elif dataname == 'tornado':
            result = np.zeros((3, 96, 96, 96))
            
        start = datetime.now()
        for x, qmap, z_ind, y_ind, x_ind in test_dataloader:
            x = x.to(device)
            qmap = qmap.to(device)
            lmbdamap = quality2lambda(qmap, a)
            # lmbdamap = qmap
            out_net = model(x, qmap)
            
            # for name, param in model.state_dict().items():
            #     print(name, param.size())
            out_criterion = criterion(out_net, x, lmbdamap)
            bpp, psnr = metric(out_net, x)
            if Q: # fake aux_loss for quantization with loss
                logger.update_test(bpp, psnr, out_criterion, out_criterion['loss'])
            else:
                logger.update_test(bpp, psnr, out_criterion, model.aux_loss())
            out = out_net['x_hat'].detach().cpu().numpy()
            
            # print('qmap', torch.min(lmbdamap), torch.max(lmbdamap) )
            # print('x', torch.min(x), torch.max(x))
            # print('out_net', torch.min(out_net['x_hat']), torch.max(out_net['x_hat']))

            for ii in range(len(z_ind)):
                if len(result.shape) == 3:
                    if padding > 0:
                        result[int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += out[ii].reshape(blocksize, blocksize, blocksize)[padding:-padding, padding:-padding, padding:-padding]
                    else:
                        result[int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += out[ii].reshape(blocksize, blocksize, blocksize)
                else:
                    if padding > 0:
                        result[:,int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += out[ii].reshape(-1, blocksize, blocksize, blocksize)[:, padding:-padding, padding:-padding, padding:-padding]
                    else:
                        result[:,int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += out[ii].reshape(-1, blocksize, blocksize, blocksize)
        
        del x, qmap, z_ind, y_ind, x_ind, out_net, out_criterion, lmbdamap
        
        h, m, s = get_duration(start, datetime.now())
        time_str = 'Test Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)  
        print(time_str)
        
        
        loss.update(logger.loss.avg)
        bpp_loss.update(logger.bpp_loss.avg)
        mse_loss.update(logger.mse_loss.avg)
        
        if dataname == 'isabel':
            result = denormalize_max_min(result, min_max_pressure).reshape(-1)
            # result = denormalize_zscore(result, mean_std_pressure).reshape(-1)
        elif dataname == 'vortex':
            result = denormalize_max_min(result, min_max_vortex).reshape(-1)
        elif dataname == 'combustion':
            result = denormalize_max_min(result, min_max_combustion_vort).reshape(-1)
        elif dataname == 'nyx':
            result = denormalize_max_min(result, min_max_nyx).reshape(-1)
        elif dataname == 'tornado':
            result[0] = denormalize_max_min(result[0], min_max_Tornado_u)
            result[1] = denormalize_max_min(result[1], min_max_Tornado_v)
            result[2] = denormalize_max_min(result[2], min_max_Tornado_w)
            
            result = result.reshape(3,-1).T
            result = result.reshape(-1)
            
        with open(os.path.join(logger.output_dir, dataname[:3]+f'_bpp{logger.bpp_loss.avg:.4f}_mse{logger.mse_loss.avg:.4f}_loss{logger.loss.avg:.4f}.bin'), 'wb') as f:
            f.write(struct.pack('<%df' % len(result), *result))
        
        # del all_data, result
        # print(
        #     f' BPP all: {bpp_all:.4f} |'
        #     f' PSNR all: {psnr_all:.4f} |'
        #     f' MS SSIM all: {ms_ssim_all:.4f}')
        print(f'[ Test ] Total mean: {loss.avg:.4f}')
        
    logger.init()
    model.train()

    return loss.avg, bpp_loss.avg, mse_loss.avg


def train(args, config, base_dir, snapshot_dir, output_dir, log_dir):
    device = torch.device("cuda" if config['cuda'] else "cpu") 
    # device = 'cpu'
    print('device', device)
    
    criterion = PixelwiseRateDistortionLoss(config['beta'], block_size=config['blocksize']-2*config['padding'], dataname=config['dataname'])
    metric = get_metric(dataname=config['dataname'], block_size=config['blocksize']-2*config['padding'])
    train_dataloader, test_dataloader = get_train_test_dataloader(config)
    logger = Logger(config, base_dir, snapshot_dir, output_dir, log_dir)

    # model = MySpatiallyAdaptiveCompression_lighter(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'], inmap=config['inmap'], ind=config['ind'])
    model = MySpatiallyAdaptiveCompression(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'], inmap=config['inmap'], ind=config['ind'])
    # model = MySpatiallyAdaptiveCompressionSingleAE(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'])
    # model = MySpatiallyAdaptiveCompression_no_UpsampledCond(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=config['prior_nc'], inmap=config['inmap'], ind=config['ind'])
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    aux_optimizer = optim.Adam(model.aux_parameters(), lr=config['lr_aux'])

    if args.resume:
        itr, model = load_checkpoint(args.resume, model, optimizer, aux_optimizer)
        logger.load_itr(itr)

    if config['set_lr']:
        lr_prior = optimizer.param_groups[0]['lr']
        for g in optimizer.param_groups:
            g['lr'] = float(config['set_lr'])
        print(f'[set lr] {lr_prior} -> {optimizer.param_groups[0]["lr"]}')

    
    # profiling
    # prof = torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/myAEmodel'),
    #     record_shapes=True,
    #     with_stack=True)
    # prof.start()
    a = config['a']
    # b = config['b']

    model.train()
    loss_best = 1e10
    while logger.itr < config['max_itr']:
        # print('logger.itr', logger.itr)
        for x, qmap in train_dataloader:
            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            x = x.to(device)
            qmap = qmap.to(device)
            lmbdamap = quality2lambda(qmap, a)
            # lmbdamap = qmap

            out_net = model(x, qmap)
            
            out_criterion = criterion(out_net, x, lmbdamap)
            
            out_criterion['loss'].backward()
            aux_loss = model.aux_loss()
            aux_loss.backward()
            # for stability
            if out_criterion['loss'].isnan().any() or out_criterion['loss'].isinf().any() or out_criterion['loss'] > 10000:
                continue

            # if config['clip_max_norm'] > 0:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_max_norm'])
            optimizer.step()
            aux_optimizer.step()  # update quantiles of entropy bottleneck modules

            # print('profing:', logger.itr)
            # prof.step()
        

        # logging
        logger.update(out_criterion, aux_loss)
        if logger.itr % config['log_itr'] == 0:
            logger.print()
            logger.write()
            logger.init()
           
        # test and save model snapshot
        if logger.itr % config['test_itr'] == 0 or logger.itr % config['snapshot_save_itr'] == 0:
            model.update()
            loss, bpp_loss, mse_loss = test(config['dataname'], logger, test_dataloader, model, criterion, metric, 
                                            val_path=config['test_data_file_path'], blocksize=config['blocksize'], padding=config['padding'],
                                            a=a)
            # model.train()
            if loss < loss_best:
                print('Best!')
                save_checkpoint(os.path.join(snapshot_dir, 'best.pt'), logger.itr, model, optimizer, aux_optimizer)
                loss_best = loss
            if logger.itr % config['snapshot_save_itr'] == 0:
                save_checkpoint(os.path.join(snapshot_dir, f'{logger.itr:05}_{bpp_loss:.4f}_{mse_loss:.8f}.pt'),
                                logger.itr, model, optimizer, aux_optimizer)

        # lr scheduling
        if logger.itr % config['lr_shedule_step'] == 0:
            lr_prior = optimizer.param_groups[0]['lr']
            for g in optimizer.param_groups:
                g['lr'] *= config['lr_shedule_scale']
            print(f'[lr scheduling] {lr_prior} -> {optimizer.param_groups[0]["lr"]}')

    del x, qmap, out_net, lmbdamap


def main(argv):
    args = parse_args(argv)
    config, base_dir, snapshot_dir, output_dir, log_dir = init(args)
    if args.seed is not None:
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU

    print('[PID]', os.getpid())
    print('[config]', args.config)
    msg = f'======================= {args.name} ======================='
    print(msg)
    for k, v in config.items():
        if k in {'lr', 'set_lr', 'p'}:
            print(f' *{k}: ', v)
        else:
            print(f'  {k}: ', v)
    print('=' * len(msg))
    print()

    if args.train:
        train(args, config, base_dir, snapshot_dir, output_dir, log_dir)
        # train_withoutQ(args, config, base_dir, snapshot_dir, output_dir, log_dir)

if __name__ == '__main__':
    main(sys.argv[1:])
