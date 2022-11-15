from numpy.testing._private.utils import decorate_methods
import pandas as pd
import numpy as np
import random
from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import hflip, to_tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from utils import read_gzip_data_noland, read_vortex_data, read_combustion_data, read_tornado_data, read_nyx_data, read_data_bin_pad

import struct

min_max_pressure = {
	'min': -4696.609,
	'max': 2102.8174
}

# min_max_pressure = { #P05.bin
# 	'min': -5471.85791, 
# 	'max': 3225.42578
# }

mean_std_pressure = {
	'mean': 315.0731,
	'std': 91.3699
}

min_max_vortex = {
    'min': 0.005305924, 
	'max': 10.520865
}

min_max_combustion_vort = {
    'min': 4.7771946e-11, 
    'max': 1573602.6
}

min_max_nyx = {
    'min': 8.773703, 
    'max': 12.799037
}


min_max_Tornado_u = {
    'min': -0.3099282, 
    'max': 0.30854315
}

min_max_Tornado_v = {
    'min': -0.30439323, 
    'max': 0.3089567
}

min_max_Tornado_w = {
    'min':-0.0016081004, 
    'max': 0.2849903
}


def normalize_zscore(d, mean_std):
    return (d - mean_std['mean']) / mean_std['std']

def denormalize_zscore(d, mean_std):
    return d * mean_std['std'] + mean_std['mean']
 
def normalize_max_min(d, min_max):
    return (d - min_max['min']) / (min_max['max'] - min_max['min'])

def denormalize_max_min(d, min_max):
    return d * (min_max['max'] - min_max['min']) + min_max['min']

# conver distance to probability
def dist_prob_reverse(dist, alpha=1., c=0.5):
    return 1.0 / (np.power(np.abs(dist), alpha) + c)

def dist_prob_exp(dist, beta=1., alpha=1.):
    return alpha * np.exp(-beta*np.abs(dist))

def dist_prob_relu(dist, C=2., th=1.):
    return np.array([C-np.abs(d) if np.abs(d)<=th else 0 for d in dist])


class MyQualityMapDataset(Dataset):
    def __init__(self, dataname, data_file_path, data_map_file_path, data_file_path_txt=None, blocksize=24, mode='train', level_range=(0, 100), uniform_value=1., p=0.2, padding=4):
        self.blocksize = blocksize
        self.mode = mode
        self.padding = padding
        self.level_range = level_range
        self.p = p
        self.data = None
        self.dataname = dataname
        self.grid = self._get_grid((blocksize, blocksize, blocksize))
        self.data_file_path_txt = None
        self.data_file_path = None

        if self.dataname == 'isabel':
            self.size = [96, 512, 512] #z, y, x, channels 
        elif self.dataname == 'vortex':
            self.size = [128, 128, 128]
        elif self.dataname == 'combustion':
            self.size = [128, 720, 480]
        elif self.dataname == 'nyx' or self.dataname == 'nyx_ensemble':
            self.size = [256, 256, 256]
        elif self.dataname == 'tornado':
            self.size = [96, 96, 96, 3]

        if data_file_path:
            self.data_file_path = data_file_path
            if self.dataname == 'isabel':
                self.data, self.filename = read_gzip_data_noland(data_file_path, padding=self.padding)
            elif self.dataname == 'vortex':
                self.data, self.filename = read_vortex_data(data_file_path, padding=self.padding)
            elif self.dataname == 'combustion':
                self.data, self.filename = read_combustion_data(data_file_path, padding=self.padding)
            elif self.dataname == 'nyx' or self.dataname == 'nyx_ensemble':
                self.data, self.filename = read_nyx_data(data_file_path, padding=self.padding)
                if self.dataname == 'nyx_ensemble':
                    self.data = np.log10(self.data)
            elif self.dataname == 'tornado':
                self.data, self.filename = read_tornado_data(data_file_path, padding=self.padding)
            
        if data_file_path_txt:
            self.data_file_path_txt = data_file_path_txt
            fh = open(data_file_path_txt, 'r')
            self.data_files = []
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                self.data_files.append(line)
            fh.close()
        
        if data_map_file_path:
            self.data_map = read_data_bin_pad(data_map_file_path, self.size, padding=self.padding)
            self.data_map = self.data_map.reshape(-1)
            print(np.min(self.data_map), np.max(self.data_map))
            
            # self.data_map = np.array([2-np.abs(d) if np.abs(d)<=1 else 0 for d in self.data_map])
            # self.data_map = dist_prob_reverse(self.data_map)
	    
            # only use when IMP is isosurface_distance map
            self.data_map = dist_prob_exp(self.data_map, beta=0.2, alpha=1.) # use this # 0.2, 1
            # only use when IMP is isosurface_distance map
            
            print(np.min(self.data_map), np.max(self.data_map))
            self.data_map.shape = self.data.shape
        
        elif self.data is not None and len(self.size)==3:
            # self.data_map = np.array(np.gradient(self.data)).reshape(3, -1)
            # self.data_map = np.linalg.norm(self.data_map, axis=0)
            # self.data_map = (self.data_map - np.min(self.data_map)) / (np.max(self.data_map) - np.min(self.data_map))
            # self.data_map.shape = self.data.shape
            self.data_map = np.zeros((self.data.shape)) + uniform_value
            print('test uniform map')
            print(np.min(self.data_map), np.max(self.data_map))
        
        elif self.data is not None and len(self.size)==4:
            self.data_map = np.zeros((self.data.shape[1:])) + uniform_value
            print(np.min(self.data_map), np.max(self.data_map))
        
        self.x_cnt = int(self.size[2]/(blocksize-2*self.padding))  
        self.y_cnt = int(self.size[1]/(blocksize-2*self.padding)) 
        self.z_cnt = int(self.size[0]/(blocksize-2*self.padding))

        if self.data_file_path_txt and self.mode == 'train':
            self.data_length = len(self.data_files)
        elif self.data_file_path and self.mode == 'train':
            self.data_length = 600
        else:
            self.data_length = self.x_cnt * self.y_cnt * self.z_cnt
        print(f'[{mode}set] {self.data_length} blocks for quality {uniform_value}')
        
    def __len__(self):
        return self.data_length
        
    def _get_grid(self, size):
        x1 = torch.tensor(range(size[0]))
        x2 = torch.tensor(range(size[1]))
        x3 = torch.tensor(range(size[2]))
        grid_x1, grid_x2, grid_x3 = torch.meshgrid(x1, x2, x3)

        grid1 = grid_x1.view(size[0], size[1], size[2], 1)
        grid2 = grid_x2.view(size[0], size[1], size[2], 1)
        grid3 = grid_x3.view(size[0], size[1], size[2], 1)
        grid = torch.cat([grid1, grid2, grid3], dim=-1)
        return grid

    def __getitem__(self, idx):
        if self.mode == 'train' and self.data_file_path:
            ii = random.randint(0, self.size[2] + 2*self.padding - self.blocksize)
            jj = random.randint(0, self.size[1] + 2*self.padding - self.blocksize)
            kk = random.randint(0, self.size[0] + 2*self.padding - self.blocksize)
            if len(self.size) == 3:
                block_d = self.data[kk:kk+self.blocksize, jj:jj+self.blocksize, ii:ii+self.blocksize]
                block_map = self.data_map[kk:kk+self.blocksize, jj:jj+self.blocksize, ii:ii+self.blocksize]
            elif len(self.size) == 4:
                block_d = self.data[:, kk:kk+self.blocksize, jj:jj+self.blocksize, ii:ii+self.blocksize]
                block_map = self.data_map[kk:kk+self.blocksize, jj:jj+self.blocksize, ii:ii+self.blocksize]
                
        elif self.mode == 'train' and self.data_file_path_txt:
            if len(self.size)==3:
                block_d = np.fromfile(self.data_files[idx], '<f4').copy().reshape(self.blocksize, self.blocksize, self.blocksize)
                block_map = np.array(np.gradient(block_d)).reshape(3, -1)
                block_map =  np.linalg.norm(block_map, axis=0)
                block_map = (block_map - np.min(block_map)) / (np.max(block_map) - np.min(block_map))
                block_map.shape = block_d.shape
            elif len(self.size) == 4:
                block_d = np.fromfile(self.data_files[idx], '<f4').copy().reshape(3, self.blocksize, self.blocksize, self.blocksize)
                block_map = np.zeros((block_d.shape[1:])) + 1.
    
        else:
            kk = int(idx / (self.x_cnt * self.y_cnt))
            jj = int((idx - kk * (self.x_cnt * self.y_cnt)) / self.x_cnt)
            ii = int(idx - kk * (self.x_cnt * self.y_cnt) - jj * self.x_cnt)
            bs = self.blocksize - 2*self.padding
            if len(self.size)==3:
                block_d = self.data[kk*bs:kk*bs+self.blocksize, jj*bs:jj*bs+self.blocksize, ii*bs:ii*bs+self.blocksize]
            elif len(self.size)==4:
                block_d = self.data[:, kk*bs:kk*bs+self.blocksize, jj*bs:jj*bs+self.blocksize, ii*bs:ii*bs+self.blocksize]
            block_map = self.data_map[kk*bs:kk*bs+self.blocksize, jj*bs:jj*bs+self.blocksize, ii*bs:ii*bs+self.blocksize]
            
        # random rate for each class
        block_map = np.array(block_map)
        qmap = np.zeros_like(block_map, dtype=float)
        uniques = np.unique(block_map)
        
        if self.mode == 'train':
            sample = random.random()
            if sample < self.p:
                if random.random() < 0.01:
                    qmap[:] = 0
                else:
                    qmap[:] = random.random()
	 
            elif sample < 2 * self.p:
                for v in uniques:
                    level = random.random()
                    qmap[block_map == v] = level
			
            elif sample < 3 * self.p:
                # gradation between horizontal or vertical
                v1 = random.random()
                v2 = random.random() 
                if random.random() < 0.5:
                    qmap = np.tile(np.linspace(v1, v2, self.blocksize), (self.blocksize, 1)).astype(float)
                    qmap = np.tile(qmap, (self.blocksize, 1, 1))
                else:
                    qmap = np.tile(np.linspace(v1, v2, self.blocksize), (self.blocksize, 1)).astype(float)
                    qmap = np.tile(qmap.T, (self.blocksize)).reshape(self.blocksize, self.blocksize, self.blocksize) 
                    
            elif sample < 4 * self.p:
                # gaussian kernel
                gaussian_num = int(1 + random.random() * 20)
                for i in range(gaussian_num):
                    mu_x = self.blocksize * random.random()
                    mu_y = self.blocksize * random.random()
                    mu_z = self.blocksize * random.random()
                    var_x = 2000 * random.random() + 1000
                    var_y = 2000 * random.random() + 1000
                    var_z = 2000 * random.random() + 1000

                    m = MultivariateNormal(torch.tensor([mu_x, mu_y, mu_z]), torch.tensor([[var_x, 0, 0], [0, var_y, 0], [0, 0, var_z]]))
                    p = m.log_prob(self.grid)
                    kernel = torch.exp(p).numpy()
                    qmap += kernel
                qmap *= 1. / qmap.max() * (0.5 * random.random() + 0.5)
		
            else:
		# input map
                qmap = block_map
        else:
            qmap = block_map
        
        if self.dataname == 'isabel':
            block_d = normalize_max_min(block_d, min_max_pressure).reshape(1, self.blocksize, self.blocksize, self.blocksize)
            # block_d = normalize_zscore(block_d, mean_std_pressure).reshape(1, self.blocksize, self.blocksize, self.blocksize)
        elif self.dataname == 'vortex':
            block_d = normalize_max_min(block_d, min_max_vortex).reshape(1, self.blocksize, self.blocksize, self.blocksize)
        elif self.dataname == 'combustion':
            block_d = normalize_max_min(block_d, min_max_combustion_vort).reshape(1, self.blocksize, self.blocksize, self.blocksize)
        elif self.dataname == 'nyx' or self.dataname == 'nyx_ensemble':
            block_d = normalize_max_min(block_d, min_max_nyx).reshape(1, self.blocksize, self.blocksize, self.blocksize)
        elif self.dataname == 'tornado':
            block_d = block_d.reshape(3, self.blocksize, self.blocksize, self.blocksize)
            tmp0 = normalize_max_min(block_d[0], min_max_Tornado_u)
            tmp1 = normalize_max_min(block_d[1], min_max_Tornado_v)
            tmp2 = normalize_max_min(block_d[2], min_max_Tornado_w)
            block_d = np.stack([tmp0, tmp1, tmp2], axis=0)
        
        block_d = torch.from_numpy(block_d.astype(np.float32))
        qmap = torch.from_numpy(qmap.astype(np.float32)).unsqueeze(dim=0)
        if self.mode == 'train':
            return block_d, qmap # training
        return block_d, qmap, kk, jj, ii # testing


# class MyQualityMapDataset_latents(Dataset):
#     def __init__(self, dataname, data_file_path, data_map_file_path, blocksize=24):
#         self.blocksize = blocksize
#         self.data = None
#         self.dataname = dataname
#         self.data_length = 500

#         if data_file_path:
#             if self.dataname == 'isabel':
#                 self.data, self.filename = read_gzip_data_noland(data_file_path)
#                 self.size = [96, 512, 512]
#             elif self.dataname == 'vortex':
#                 self.data, self.filename = read_vortex_data(data_file_path)
#                 self.size = [128, 128, 128]
#             elif self.dataname == 'combustion':
#                 self.data, self.filename = read_combustion_data(data_file_path)
#                 self.size = [480, 720, 128]
        
#         if data_map_file_path:
#             self.data_map = read_data_bin_pad(data_map_file_path, self.size)
#             self.data_map = self.data_map.reshape(-1)
#             print(np.min(self.data_map), np.max(self.data_map))
#             # self.data_map = np.array([2-np.abs(d) if np.abs(d)<=1 else 0 for d in self.data_map])
#             # self.data_map = dist_prob_reverse(self.data_map)
#             self.data_map = dist_prob_exp(self.data_map, beta=0.2, alpha=2.)
#             print(np.min(self.data_map), np.max(self.data_map))
#             self.data_map.shape = self.data.shape
#         else:
#             self.data_map = np.array(np.gradient(self.data)).reshape(3, -1)
#             self.data_map = 2. - np.linalg.norm(self.data_map, axis=0)
#             self.data_map.shape = self.data.shape
            
#         self.rand = False
#         if not self.rand:
#             np.random.seed(0)
#             self.locations_x = np.random.randint(self.size[2] + 2*4 - self.blocksize, size=self.data_length)
#             self.locations_y = np.random.randint(self.size[1] + 2*4 - self.blocksize, size=self.data_length)
#             self.locations_z = np.random.randint(self.size[0] + 2*4 - self.blocksize, size=self.data_length)
              
#         print(f'{self.data_length} blocks')
        
#     def __len__(self):
#         return self.data_length

#     def __getitem__(self, idx):
#         if self.rand:
#             ii = random.randint(0, self.size[2] + 2*4 - self.blocksize)
#             jj = random.randint(0, self.size[1] + 2*4 - self.blocksize)
#             kk = random.randint(0, self.size[0] + 2*4 - self.blocksize)
#         else:
#             ii = self.locations_x[idx]
#             jj = self.locations_y[idx]
#             kk = self.locations_z[idx]
            
#         block_d = self.data[kk:kk+self.blocksize, jj:jj+self.blocksize, ii:ii+self.blocksize]
#         block_map = self.data_map[kk:kk+self.blocksize, jj:jj+self.blocksize, ii:ii+self.blocksize]
        
#         if self.dataname == 'isabel':
#             block_d = normalize_zscore(block_d, mean_std_pressure).reshape(1, self.blocksize, self.blocksize, self.blocksize)
#         elif self.dataname == 'vortex':
#             block_d = normalize_max_min(block_d, min_max_vortex).reshape(1, self.blocksize, self.blocksize, self.blocksize)
#         elif self.dataname == 'combustion':
#             block_d = normalize_max_min(block_d, min_max_combustion_vort).reshape(1, self.blocksize, self.blocksize, self.blocksize)
        
#         block_d = torch.from_numpy(block_d.astype(np.float32))
#         qmap = torch.from_numpy(block_map.astype(np.float32)).unsqueeze(dim=0)
#         return block_d, qmap, kk, jj, ii


def blockshaped(arr, nz, ny, nx): 
    d, h, w, v = arr.shape
    return (arr.reshape(d//nz, nz, h//ny, ny, w//nx, nx, v).swapaxes(3,4).swapaxes(1,3).swapaxes(1,2).reshape(-1, nz, ny, nx, v))
    

# class LatentDataset(Dataset):
#     def __init__(self, path_y, path_z, N, M, level=0, shape=None):
#         # self.latent_y = np.load(path_y).reshape(-1, 16, 3, 3, 3)
#         # self.latent_z = np.load(path_z).reshape(-1, 16, 2, 2, 2)
#         if shape is not None:
#             reduced_size = 2**(level)
#             latent_y = np.load(path_y).reshape(shape[0]*reduced_size, shape[1]*reduced_size, shape[2]*reduced_size, -1) # M, 3, 3, 3
#             latent_z = np.load(path_z).reshape(shape[0]*reduced_size, shape[1]*reduced_size, shape[2]*reduced_size, -1) # N, 2, 2, 2
            
#             self.latent_y = np.mean(blockshaped(latent_y, reduced_size, reduced_size, reduced_size).reshape(-1, reduced_size**3, M, 3, 3, 3), axis=1)
#             self.latent_z = np.mean(blockshaped(latent_z, reduced_size, reduced_size, reduced_size).reshape(-1, reduced_size**3, N, 2, 2, 2), axis=1)
#             del latent_y, latent_z
#             self.latent_y = np.array(self.latent_y)
#             self.latent_z = np.array(self.latent_z)
                        
#         else:
#             self.latent_y = np.load(path_y).reshape(-1, M, 3, 3, 3)
#             self.latent_z = np.load(path_z).reshape(-1, N, 2, 2, 2)
#         self.level = level

#     def __len__(self):
#         return len(self.latent_y) #// (2**(3*self.level))

#     def __getitem__(self, idx):
#         # print(idx,  torch.from_numpy(self.latent_y[idx].astype(np.float32)).shape)
#         return torch.from_numpy(self.latent_y[idx].astype(np.float32)), torch.from_numpy(self.latent_z[idx].astype(np.float32))


def get_train_test_dataloader(config):
    train_dataset = MyQualityMapDataset(config['dataname'], data_file_path=config['train_data_file_path'], data_map_file_path=config['data_map_file_path'], data_file_path_txt=config['data_file_path_txt'],
                                        blocksize=config['blocksize'], mode='train', p=config['p'], padding=config['padding'])
    train_dataloader = DataLoader(train_dataset, batch_size=config['batchsize'], shuffle=True)
    
    test_dataset = MyQualityMapDataset(config['dataname'], data_file_path=config['test_data_file_path'], data_map_file_path=config['data_map_file_path'],
                                          blocksize=config['blocksize'], mode='test', p=config['p'], padding=config['padding'])
    test_dataloader = DataLoader(test_dataset, batch_size=config['batchsize_test'], shuffle=False)
    return train_dataloader, test_dataloader


def get_testdataloader(config, uniform_value=1.):
    # levels = [-100] + [int(100*(i/L)) for i in range(L+1)]
    levels = [0]
    test_dataloaders = []
    for level in levels:
        test_dataset = MyQualityMapDataset(config['dataname'], data_file_path=config['test_data_file_path'], data_map_file_path=config['data_map_file_path'], blocksize=config['blocksize'], mode='test', p=config['p'], uniform_value=uniform_value, padding=config['padding'])
        test_dataloader = DataLoader(test_dataset, batch_size=config['batchsize_test'], shuffle=False)
        test_dataloaders.append(test_dataloader)
    return test_dataloaders


# def get_latentloader(config, level=0):
#     reduced_size = 2**(level)
#     act_size = config['blocksize'] - 2 * config['padding']
#     if config['dataname'] == 'isabel':
#         shape = np.array([96//reduced_size, 512//reduced_size, 512//reduced_size])
#     elif config['dataname'] == 'vortex':
#         shape = np.array([128//reduced_size, 128//reduced_size, 128//reduced_size])
#     elif config['dataname'] == 'combustion':
#         shape = np.array([128//reduced_size, 720//reduced_size, 480//reduced_size])
#     elif config['dataname'] == 'nyx' or dataname == 'nyx_ensemble':
#         shape = np.array([256//reduced_size, 256//reduced_size, 256//reduced_size])
#     elif config['dataname'] == 'tornado':
#         shape = np.array([96//reduced_size, 96//reduced_size, 96//reduced_size])
    
#     shape = shape // act_size
#     latent_dataset = LatentDataset(config['laty_path'], config['latz_path'], N=config['N'], M=config['M'], level=level, shape=shape)
#     latent_dataloader = DataLoader(latent_dataset, batch_size=config['batchsize_test'], shuffle=False,
#                                  num_workers=config['worker_num'])
#     return latent_dataloader

