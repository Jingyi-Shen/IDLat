import json
from pprint import pprint
import pstats
import numpy as np
import os
import struct

from utils import read_gzip_data_noland, read_vortex_data, read_combustion_data, read_data_bin_pad, read_nyx_data, read_tornado_data

import pdb

def fromParaview(jsonfile):
    # How to save transfer function from Paraview
    # In 'color mapp editor', click the button 'Save to present' (a folder icon with a green arrow)
    # A window 'Save present option' will pop out, make sure 'Save opatities' is checked. Then click 'OK'
    # A window 'Choose present' will pop out, click 'Export' and set the transfer function to a .json file

    # # load transfer function from paraview
    # paraview transfer function assume the data range are 0 to 1.
    # when you setup the trasnfer function in vtk, you have to rescale it to the data range of the dataset
    json_data=open(jsonfile)
    data = json.load(json_data)
    json_data.close()
    # pprint(data)

    nOpaPointVs = []
    nOpaPointOpas = []
    nOpaPoint = int( len( data[0]['Points'])/4 ) # number of opacity function control point
    for i in range( nOpaPoint ):
        dtValue = data[0]['Points'][i*4]
        opaValue = data[0]['Points'][i*4+1]
        print('opacity control point: ', i, ': ', dtValue, opaValue)
        nOpaPointVs.append(dtValue)
        nOpaPointOpas.append(opaValue)
    
    # nRgbPoint= int( len( data[0]['RGBPoints'] ) / 4 ) # number of the color map control point
    # for i in range( nRgbPoint ):
    #     dtValue = data[0]['RGBPoints'][i*4]
    #     r = data[0]['RGBPoints'][i*4+1]
    #     g = data[0]['RGBPoints'][i*4+2]
    #     b = data[0]['RGBPoints'][i*4+3]
        # print('rgb control point: ', i, ': ', dtValue, r, g, b)

    # after load the control points from opacity function and color map, 
    # You can use them to setup you transfer function in VTK

    # now, from transfer function (opacity) to importance map in range(0, 2)
    # for every value, decide its importance (opacity)
    return nOpaPointVs, nOpaPointOpas

def load_data(data_file_path, dataname='vortex'):
    if dataname == 'isabel':
        data, filename = read_gzip_data_noland(data_file_path, padding=0)
        size = [96, 512, 512]
    elif dataname == 'vortex':
        data, filename = read_vortex_data(data_file_path, padding=0)
        size = [128, 128, 128]
    elif dataname == 'combustion':
        data, filename = read_combustion_data(data_file_path, padding=0)
        size = [480, 720, 128]
    elif dataname == 'nyx':
        data, filename = read_nyx_data(data_file_path, padding=0)
        size = [256, 256, 256]
    elif dataname == 'tornado':
        data, filename = read_tornado_data(data_file_path, padding=0)
        size = [96, 96, 96]    
    return data, size


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def segfunction(d, xs, ys):  
    #      /\
    # ____/  \___/\___
    # print(xs)
    # import pdb
    # pdb.set_trace()
    if d <= xs[0]:
        return ys[0]
    elif d <= xs[-1]:
        for i in range(len(xs)-1):
            if xs[i] <= d <= xs[i+1]:
                w = (d - xs[i]) / (xs[i+1] - xs[i])
                # print(ys[i]*w + ys[i+1]*(1-w), ys[i], ys[i+1], w)
                return ys[i]*(1-w) + ys[i+1]*w #need double check

    else:
        return ys[-1]


def TF_based(smooth=True):
    dataname='vortex'
    output_dir = './data/opacity/'
    data_file_path = './data/vorts10.bin'
    ts = 10
    nOpaPointVs, nOpaPointOpas = fromParaview('./data/opacity/feature.json')
    data, size = load_data(data_file_path, dataname)
    data = data.reshape(-1)
    data_map = np.zeros((size)).reshape(-1)

    for i in range(len(data_map)):
        data_map[i] = segfunction(data[i], nOpaPointVs, nOpaPointOpas)
        # if i > 1000:
        #     exit()
    # save into bin
    # import pdb
    # pdb.set_trace()
    if smooth:
        # box = np.ones((3,3,3))/27. # only used for 1D
        # data_map = np.convolve(data_map, box, mode='same')
        from scipy.ndimage import gaussian_filter
        data_map = gaussian_filter(data_map, sigma=1)
    
    data_map = np.array(data_map).reshape(-1)
    with open(os.path.join(output_dir+'importance/', 'map_'+dataname[:3]+f'_{ts:03}_TFfeat_smoothed.bin'), 'wb') as f:
        f.write(struct.pack('<%df' % len(data_map), *data_map))

def gradient_based():
    dataname='vortex'
    output_dir = './data/opacity/'
    data_file_path = './data/vorts10.bin'
    ts = 10
    data, size = load_data(data_file_path, dataname)
    # maps = np.zeros((size)).reshape(-1)
    data_map = np.array(np.gradient(data)).reshape(3, -1)
    data_map = np.linalg.norm(data_map, axis=0)
    # normalize to 0-1
    data_map = 1-(data_map - np.min(data_map)) / (np.max(data_map) - np.min(data_map))
    data_map = np.array(data_map*2).reshape(-1)
    with open(os.path.join(output_dir+'importance/', 'map_'+dataname[:3]+f'_{ts:03}_revGradx2.bin'), 'wb') as f:
        f.write(struct.pack('<%df' % len(data_map), *data_map))


def value_based(threshold=10.5):
    dataname='vortex'
    if dataname=='nyx':
        output_dir = './data/nyx_high_density_map_ensemble/'
        data_file_dir = '/fs/project/PAS0027/nyx/256/output/'
        param_dirs = np.array([f for f in sorted(os.listdir(data_file_dir)) ])[:800]
        files = np.array([f+'/Raw_plt256_00200/density.bin' for f in  param_dirs])
        import pdb
        # parameters = []
        for idx in range(800):
            # data_file_path = '/fs/project/PAS0027/nyx/256/256CombineFiles/raw/%d.bin' % (ts)
            # data_file_path = './data/nyx0.bin'
            # print(idx, files[idx])
            data, size = load_data(data_file_dir+files[idx], dataname)
            data = np.log10(data)
            map_name = files[idx].split('/')[0]
            maps = (data>threshold)
            maps = maps.astype(float)
            maps = maps.reshape(-1)
            print(idx, np.min(maps), np.max(maps), np.min(data), np.max(data))
            # pdb.set_trace()
            with open(os.path.join(output_dir, 'map_'+map_name+f'_thresh10_5.bin'), 'wb') as f:
                f.write(struct.pack('<%df' % len(maps), *maps))
    
    elif dataname=='vortex':
        # output_dir = './data/vortex_iso_value/'
        output_dir = './data/vortex_iso_value_v3_01/'
        #  /fs/project/PAS0027/vortex_data/vortex/vorts01.data
        data_file_dir = '/fs/project/PAS0027/vortex_data/vortex/'
        files = np.array([f for f in sorted(os.listdir(data_file_dir)) if f.endswith('.data')])
        print(files)
        
        iso_values = np.linspace(2.0, 9.9, 80) #4.0~9.9
        # iso_values = np.linspace(2.0, 9.0, 8) #4.0~9.9
        
        # from scipy.ndimage import gaussian_filter
        
        for idx in range(len(files)):
            # data_file_path = '/fs/project/PAS0027/nyx/256/256CombineFiles/raw/%d.bin' % (ts)
            # data_file_path = './data/nyx0.bin'
            print(idx, files[idx])
            data, size = load_data(data_file_dir+files[idx], dataname)
            map_name = files[idx].split('.')[0][-2:]
            
            for iso in iso_values:
                min_ = iso - 0.1
                max_ = iso + 0.1
                # maps = (data>=min_) & (data<=max_)
                maps = (data>min_) & (data<max_)
                maps = maps.astype(float)
                
                # import pdb
                # pdb.set_trace()
                # # appy blur
                # from scipy import signal
                # # build the smoothing kernel
                # sigma = 1.0     # width of kernel
                # x = np.arange(-1,2,1)   # coordinate arrays -- make sure they contain 0!
                # y = np.arange(-1,2,1)
                # z = np.arange(-1,2,1)
                # xx, yy, zz = np.meshgrid(x,y,z)
                # kernel = 1./(2.* np.pi * sigma**2) * np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
                # kernel = kernel / np.sum(kernel)
                # maps = signal.convolve(maps, kernel, mode="same")
                # maps = gaussian_filter(maps, sigma=1)
                # maps = (maps - np.min(maps)) / (np.max(maps) - np.min(maps))
                maps = maps.reshape(-1)
                
                print(idx, iso, np.sum(maps), np.min(maps),  np.max(maps), np.min(data), np.max(data))

                with open(os.path.join(output_dir, 'map_'+map_name+'_iso%s.bin'%("{:.1f}".format(iso)) ), 'wb') as f:
                    f.write(struct.pack('<%df' % len(maps), *maps))
            exit()
    



def user_selected():
    pass

def test():
    y_feat = np.fromfile('./results/vortex/AEout/best_ch16_onlyQmap/recon_50mse/vor_010_bpp0.0704_psnr17.2383_iso6_z_hat_TFfeat.bin', '<f4').copy()
    y_tf00 = np.fromfile('./results/vortex/AEout/best_ch16_onlyQmap/recon_50mse/vor_010_bpp0.0704_psnr30.7974_iso6_z_hat.bin', '<f4').copy()
    # 8*8*8*432
    import pdb
    import seaborn as sns
    import matplotlib.pyplot as plt
    pdb.set_trace()
    sns.distplot(y_feat, kde=False, hist=True, hist_kws={"range": [-5,11]}, label='feature')
    sns.distplot(y_tf00, kde=False, hist=True, hist_kws={"range": [-5,11]}, label='1st TF')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # TF_based()
    # gradient_based()
    value_based()
