import os
import yaml
import time
import struct
import shutil
from shutil import copy2
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

import gzip
import scipy.ndimage
import numpy as np
import netCDF4 as nc


def get_config(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
        return config


def init(args):
    base_dir = f'./results/{args.name}'
    snapshot_dir = os.path.join(base_dir, 'snapshots')
    output_dir = os.path.join(base_dir, 'outputs')
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    config = get_config(args.config)
    try:
        copy2(args.config, os.path.join(base_dir, 'config.yaml'))
    except shutil.SameFileError:
        pass

    return config, base_dir, snapshot_dir, output_dir, log_dir


class AverageMeter:
    """Compute running average."""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    def __init__(self, config, base_dir, snapshot_dir, output_dir, log_dir, level_num=11, only_print=False):
        self.config = config
        self.base_dir = base_dir
        self.snapshot_dir = snapshot_dir
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.level_num = level_num
        self.itr = 0
        self.init()

        if not only_print:
            self._init_summary_writers(level_num)

    def _init_summary_writers(self, level_num):
        self.writer = SummaryWriter(self.log_dir)
        # self.test_writers = [SummaryWriter(os.path.join(self.log_dir, f'level_{i}')) for i in range(level_num + 1)]

    def init(self):
        self.loss = AverageMeter()
        self.bpp_loss = AverageMeter()
        self.mse_loss = AverageMeter()
        self.psnr = AverageMeter()
        # self.ms_ssim = AverageMeter()
        self.aux_loss = AverageMeter()

    def load_itr(self, itr):
        self.itr = itr

    def update(self, out_criterion, aux_loss):
        self.loss.update(out_criterion['loss'].item())
        self.bpp_loss.update(out_criterion['bpp_loss'].item())
        self.mse_loss.update(out_criterion['mse_loss'].item())
        self.aux_loss.update(aux_loss.item())
        self.itr += 1

    def update_test(self, bpp, psnr, out_criterion, aux_loss):
        self.loss.update(out_criterion['loss'].item())
        self.bpp_loss.update(bpp.item())
        self.mse_loss.update(out_criterion['mse_loss'].item())
        self.psnr.update(psnr.item())
        # self.ms_ssim.update(ms_ssim.item())
        self.aux_loss.update(aux_loss.item())

    def print(self):
        print(
            f'[{self.itr:>7}]'
            f' Total: {self.loss.avg:.4f} |'
            f' BPP: {self.bpp_loss.avg:.4f} |'
            f' MSE: {self.mse_loss.avg:.6f} |'
            f' Aux: {self.aux_loss.avg:.0f}'
        )

    def print_test(self, case=-1):
        print(
            f'[ Test{case:>2} ]'
            f' Total: {self.loss.avg:.4f} |'
            f' BPP: {self.bpp_loss.avg:.4f} |'
            f' PSNR: {self.psnr.avg:.4f} |'
            # f' MS-SSIM: {self.ms_ssim.avg:.4f} |'
            f' Aux: {self.aux_loss.avg:.0f}'
        )

    def write(self):
        self.writer.add_scalar('Total loss', self.loss.avg, self.itr)
        self.writer.add_scalar('BPP loss', self.bpp_loss.avg, self.itr)
        self.writer.add_scalar('MSE loss', self.mse_loss.avg, self.itr)
        self.writer.add_scalar('Aux loss', self.aux_loss.avg, self.itr)

    def write_test(self, level=0):
        # if self.level_num == 1:
        #     writer = self.writer
        # else:
        #     writer = self.test_writers[level]
        writer = self.writer
        writer.add_scalar('[Test] Total loss', self.loss.avg, self.itr)
        writer.add_scalar('[Test] BPP', self.bpp_loss.avg, self.itr)
        writer.add_scalar('[Test] MSE loss', self.mse_loss.avg, self.itr)
        writer.add_scalar('[Test] PSNR', self.psnr.avg, self.itr)
        # writer.add_scalar('[Test] MS-SSIM', self.ms_ssim.avg, self.itr)
        writer.add_scalar('[Test] Aux loss', self.aux_loss.avg, self.itr)


def save_checkpoint(filename, itr, model, optimizer, aux_optimizer=None, scaler=None):
    snapshot = {
        'itr': itr,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'aux_optimizer': aux_optimizer.state_dict()
    }
    if aux_optimizer is not None:
        snapshot['aux_optimizer'] = aux_optimizer.state_dict()
    if scaler is not None:
        snapshot['scaler'] = scaler.state_dict()
    torch.save(snapshot, filename)


def load_checkpoint(path, model, optimizer=None, aux_optimizer=None, scaler=None, only_net=False):
    snapshot = torch.load(path, map_location='cpu')
    itr = snapshot['itr']
    print(f'Loaded from {itr} iterations')
    # print(snapshot['model'])
    model.load_state_dict(snapshot['model'])
    # import pdb
    # pdb.set_trace()
    if not only_net:
        if 'optimizer' in snapshot:
            optimizer.load_state_dict(snapshot['optimizer'])
        if 'aux_optimizer' in snapshot:
            aux_optimizer.load_state_dict(snapshot['aux_optimizer'])
        if scaler is not None and 'scaler' in snapshot:
            scaler.load_state_dict(snapshot['scaler'])

    return itr, model


###############################################################################
import compressai

metric_ids = {
    "mse": 0,
}


def inverse_dict(d):
    # We assume dict values are unique...
    assert len(d.keys()) == len(set(d.keys()))
    return {v: k for k, v in d.items()}


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size # Size in bytes of a plain file; amount of data waiting on some special files.


def load_image(filepath: str) -> Image.Image:
    return Image.open(filepath).convert("RGB")


def img2torch(img: Image.Image) -> torch.Tensor:
    return ToTensor()(img).unsqueeze(0)


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def get_header(model_name, metric, quality):
    """Format header information:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    """
    metric = metric_ids[metric]
    code = (metric << 4) | (quality - 1 & 0x0F)
    return 0, 0  # model_ids[model_name], code


def parse_header(header):
    """Read header information from 2 bytes:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    """
    model_id, code = header
    quality = (code & 0x0F) + 1
    metric = code >> 4
    return (
        model_id,  # inverse_dict(model_ids)[model_id],
        metric,  # inverse_dict(metric_ids)[metric],
        quality,
    )


# def pad(x, p=2 ** 6):
#     h, w = x.size(2), x.size(3)
#     H = (h + p - 1) // p * p
#     W = (w + p - 1) // p * p
#     padding_left = (W - w) // 2
#     padding_right = W - w - padding_left
#     padding_top = (H - h) // 2
#     padding_bottom = H - h - padding_top
#     return F.pad(
#         x,
#         (padding_left, padding_right, padding_top, padding_bottom),
#         mode="constant",
#         value=0,
#     )


# def crop(x, size):
#     H, W = x.size(2), x.size(3)
#     h, w = size
#     padding_left = (W - w) // 2
#     padding_right = W - w - padding_left
#     padding_top = (H - h) // 2
#     padding_bottom = H - h - padding_top
#     return F.pad(
#         x,
#         (-padding_left, -padding_right, -padding_top, -padding_bottom),
#         mode="constant",
#         value=0,
#     )


def _encode_single(model, x: torch.Tensor, output: str, qmap=None, metric='mse', coder='ans', quality=1, blocksize=16, verbose=False):
    compressai.set_entropy_coder(coder)
    enc_start = time.time()

    start = time.time()
    net = model
    load_time = time.time() - start

    # B, _, z, h, w = x.shape
    B = x.shape[0]
    C = x.shape[1]
    # p = 64

    with torch.no_grad():
        if qmap is None:
            out = net.compress(x)
        else:
            out = net.compress(x, qmap)
    
    # import pdb
    # pdb.set_trace()
    # shape = out["shape"]
    # header = get_header(model, metric, quality)

    with Path(output).open("wb") as f:
        # write_uchars(f, header)
        # write shape and number of encoded latents 
        # can be improved
        write_uints(f, (len(out["strings"]),)) # batch size
        # string = (b','.join(out["strings"]))
        for s in out["strings"]:
            write_uints(f, (len(s),)) # each string length for the batch list
            write_bytes(f, s)
            # print(len(s))
      
    enc_time = time.time() - enc_start
    size = filesize(output) # file size in byte
    # print(size)
    bpp = float(size) * 8 / ( B * C * blocksize * blocksize * blocksize) # file size in bit
    if verbose:
        print(
            f"{bpp:.4f} bpp |"
            f" Encoded in {enc_time:.4f}s (model loading: {load_time:.4f}s) |"
            f" file size {size:.4f}"
        )
    return out, bpp, size, enc_time


def _decode_single(model, inputpath, n_strings=2, shape=(2,2,2), coder='ans', verbose=False):
    compressai.set_entropy_coder(coder)

    dec_start = time.time()
    with Path(inputpath).open("rb") as f:
        # model_, metric, quality = parse_header(read_uchars(f, 2))
        # original_size = read_uints(f, 2)
        # shape = read_uints(f, 3)
        batch = read_uints(f, 1)[0]
        # string = f.read()
        # print(string)
        string = []
        for i in range(batch):
            length = read_uints(f, 1)[0]
            s = read_bytes(f, length)
            string.append(s)
            
    start = time.time()
    net = model
    load_time = time.time() - start

    with torch.no_grad():
        out = net.decompress(string, shape)

    # x_hat = crop(out["x_hat"], original_size)
    # x_hat.clamp_(0, 1)
    dec_time = time.time() - dec_start
    if verbose:
        print(f"Decoded in {dec_time:.4f}s (model loading: {load_time:.4f}s)")

    return out, dec_time
    

def _encode(model, x: torch.Tensor, output: str, qmap=None, metric='mse', coder='ans', quality=1, blocksize=16, verbose=False):
    compressai.set_entropy_coder(coder)
    enc_start = time.time()

    start = time.time()
    net = model
    load_time = time.time() - start

    # B, _, z, h, w = x.shape
    B = x.shape[0]
    C = x.shape[1]
    # p = 64
    # x = pad(x, p)

    with torch.no_grad():
        if qmap is None:
            out = net.compress(x)
        else:
            out = net.compress(x, qmap)
    
    # import pdb
    # pdb.set_trace()
    shape = out["shape"]
    header = get_header(model, metric, quality)

    with Path(output).open("wb") as f:
        # write_uchars(f, header)
        # write original block size
        # write_uints(f, (z, h, w))
        # write shape and number of encoded latents 
        write_uints(f, (len(out["strings"][0]),))
        for s in out["strings"]:
            # s = np.array(s).tobytes()
            # write_uints(f, (len(s),))
            for ss in s: 
                write_uints(f, (len(ss),))
                write_bytes(f, ss)

    enc_time = time.time() - enc_start
    size = filesize(output) # file size in byte
    bpp = float(size) * 8 / ( B * C * blocksize * blocksize * blocksize) # file size in bit
    if verbose:
        print(
            f"{bpp:.4f} bpp |"
            f" Encoded in {enc_time:.4f}s (model loading: {load_time:.4f}s) |"
            f" file size {size:.4f}"
        )
    return out, bpp, size, enc_time


def _decode(model, inputpath, n_strings=2, shape=(2,2,2), coder='ans', verbose=False):
    compressai.set_entropy_coder(coder)

    dec_start = time.time()
    with Path(inputpath).open("rb") as f:
        # model_, metric, quality = parse_header(read_uchars(f, 2))
        # original_size = read_uints(f, 2)
        # shape = read_uints(f, 3)
        strings = []
        batch = read_uints(f, 1)[0]
        for _ in range(n_strings):
            s = []
            # l = int(length//batch) #  wrong !
            for i in range(batch):
                length = read_uints(f, 1)[0]
                s.append(read_bytes(f, length))
            strings.append(s)
        
        # length = read_uints(f, 1)[0]
        # s = []
        # l = int(length//batch)
        # for i in range(batch):
        #     s.append(read_bytes(f, l))
        # strings.append(s)
        # # for z, a single long string
        # length = read_uints(f, 1)[0]
        # s = read_bytes(f, length)
        # strings.append(s)

    start = time.time()
    net = model
    load_time = time.time() - start

    with torch.no_grad():
        out = net.decompress(strings, shape)

    # x_hat = crop(out["x_hat"], original_size)
    # x_hat.clamp_(0, 1)
    dec_time = time.time() - dec_start
    if verbose:
        print(f"Decoded in {dec_time:.4f}s (model loading: {load_time:.4f}s)")

    return out, dec_time

# data loading 
def read_data_bin(filename):
    data = np.fromfile(filename, '<f4') 
    data = data.copy()
    data = data.reshape(-1)
    return data

def read_data_bin_pad(filename, d_shape=(128, 128, 128), padding=0):
    data = np.fromfile(filename, dtype='f').copy()
    data.shape = d_shape
    data = np.pad(data, ((padding, padding), (padding, padding), (padding, padding)), 'edge')
    return data

def read_gzip_data_noland(filename, padding=0):
    with gzip.open(filename, 'rb') as f:
        from_gzipped = np.frombuffer(f.read(), dtype='>f4')
    data = from_gzipped.copy()# # big-endian float32
    data = data.reshape(100, 500, 500)[8:,:,:]
    data = scipy.ndimage.zoom(data, [96.0/(100-8), 512.0/500, 512.0/500], order=0) # 0: nearest; 1: bilinear; 2: cubic
    # data = data.reshape(-1)
    # with open('./Pf05_512_512_96.bin', 'wb') as f:
    #     f.write(struct.pack('<%df' % len(data), *data))
    data = np.pad(data, ((padding, padding), (padding, padding), (padding, padding)), 'edge')
    name = filename.split('/')[-1].split('.')[0]
    return data, name

def read_vortex_data(filename, padding=0):
    data = np.fromfile(filename, '<f4').copy()[3:] # first 3 integers: dimension
    data = np.array(data).reshape(128, 128, 128)
    data = np.pad(data, ((padding, padding), (padding, padding), (padding, padding)), 'edge')
    name = filename.split('/')[-1].split('.')[0]
    return data, name

def read_combustion_data(filename, padding=0):
    data = np.fromfile(filename, '<f4').copy()
    data = np.array(data).reshape(120, 720, 480)
    data = scipy.ndimage.zoom(data, [128.0/120, 1., 1.], order=0)
    data = np.pad(data, ((padding, padding), (padding, padding), (padding, padding)), 'edge')
    name = filename.split('/')[-1].split('.')[0]
    return data, name

def read_nyx_data(filename, padding=0):
    data = np.fromfile(filename, '<f4').copy()
    data = np.array(data).reshape(256, 256, 256)
    data = np.pad(data, ((padding, padding), (padding, padding), (padding, padding)), 'edge')
    name = 'd_'+filename.split('/')[-1].split('.')[0]
    return data, name

def read_tornado_data(filename, padding=0):
    ds = nc.Dataset(filename)
    data = []
    name = filename.split('/')[-1].split('.')[0].split('_')[-1]
    for v in ['u','v','w']:
        d = ds[v][:].copy()
        d = np.pad(d, ((padding, padding), (padding, padding), (padding, padding)), 'edge') #symmetric, edge
        data.append(d)
    ds.close()
    data = np.array(data)
    return data, name

# time
def get_duration(start, end):
    h, remainder = divmod((end - start).seconds, 3600)
    m, s = divmod(remainder, 60)
    return h, m, s

# get from datset
def dist_prob_exp(dist, beta=1., alpha=1.):
    return alpha * np.exp(-beta*np.abs(dist))

def cal_PSNR_wMSE_wMAE(vol1, vol2, diff=100, map_path=None, size=None):
    if map_path is None:
        mse = np.mean((np.array(vol1, dtype=np.float32) - np.array(vol2, dtype=np.float32)) ** 2)
        mae = np.abs(np.array(vol1, dtype=np.float32) - np.array(vol2, dtype=np.float32)).mean()
        return 10 * np.log10(diff ** 2 / mse), mse, mae
    else:
        data_map = read_data_bin_pad(map_path, size, padding=0)
        if 'iso' in map_path:
            data_map = dist_prob_exp(data_map, beta=3, alpha=1.)  # use this
        else:
            data_map[data_map < 0.6] = 0 # do not count data quality of low importance region
        # normalize map to [0, 1] and apply weighted MSE, MAE and cal PSNR
        print(np.min(data_map), np.max(data_map))
        # data_map = (data_map - np.min(data_map)) / (np.max(data_map) - np.min(data_map))
        
        # threshold 
        data_map = data_map.reshape(-1) 
        data_map = data_map/np.sum(data_map)
        vol1 = vol1.reshape(-1)
        vol2 = vol2.reshape(-1)
        
        se = (np.array(vol1, dtype=np.float32) - np.array(vol2, dtype=np.float32)) ** 2
        w_mse = np.sum(np.multiply(se, data_map.reshape(-1)))
        ae = np.abs(np.array(vol1, dtype=np.float32) - np.array(vol2, dtype=np.float32)) 
        w_mae = np.sum(np.multiply(ae, data_map.reshape(-1)))
        return 10 * np.log10(diff ** 2 / w_mse), w_mse, w_mae

def lerp(t, v0, v1):
    '''
    Linear interpolation
    Args:
        t (float/np.ndarray): Value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    '''
    v2 = (1.0 - t) * v0 + t * v1
    return v2

def pool3d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    3D Pooling

    Parameters:
        A: input 3D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size[0])//stride[0] + 1,
                    (A.shape[1] - kernel_size[1])//stride[1] + 1,
                    (A.shape[2] - kernel_size[2])//stride[2] + 1)
    # kernel_size = (kernel_size, kernel_size, kernel_size)
    A_w = np.lib.stride_tricks.as_strided(A, shape = output_shape + kernel_size, 
                        strides = (stride[0]*A.strides[0],
                                   stride[1]*A.strides[1],
                                   stride[2]*A.strides[2]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size) #(10,10,10) --> (5,5,5); (125, 2, 2, 2)

    if pool_mode == 'max':
        return A_w.max(axis=(1,2,3)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2,3)).reshape(output_shape)
