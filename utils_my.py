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


def plot_mesh():
# 	from mesh_to_sdf import  mesh_to_voxels, sample_sdf_near_surface, mesh_to_sdf

	import trimesh
# 	import pyrender
	import skimage.measure
	import struct
	# import os
	# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# 	mesh = trimesh.load('./data/iso_mesh/vor_000_bpp0.0983_iso8.ply')

	from pysdf import SDF
	
	start = time.time()
	# Load some mesh (don't necessarily need trimesh)
	files = [f for f in os.listdir('./data/iso_mesh_cal_time/vorts01_dense/') if not f.startswith('.') and f.endswith('.ply') and 'vorts' in f]
	for fname in files:
	    
		o = trimesh.load('./data/iso_mesh_cal_time/vorts01_dense/'+fname)
		f = SDF(o.vertices, o.faces); # (num_vertices, 3) and (num_faces, 3)
# 		print(o.vertices)
		# Compute some SDF values (negative outside);
		# takes a (num_points, 3) array, converts automatically
		x = np.linspace(0, 127, 128)
		xyz = np.meshgrid(x, x, x)
		positions = np.vstack(map(np.ravel, xyz)).T
		positions = positions[:,[2,0,1]] #(x,y,z)
# 		import pdb
# 		pdb.set_trace()
		sdf_multi_point = f(positions)
		name = fname[-7:-4] #fname.split('.')[0]
		print(np.min(sdf_multi_point), np.max(sdf_multi_point), name)
# 		with open(os.path.join('./data/iso_bin/tmp/%s.bin' % name), 'wb') as f:
# 			f.write(struct.pack('<%df' % len(sdf_multi_point), *sdf_multi_point)) 
	end = time.time()
	print(end-start, 'seconds.')
	
	# voxels = mesh_to_voxels(mesh, 128, pad=False, scan_count=100, scan_resolution=400, sign_method='normal',) # a numpy array of shape (N, N, N)
	# print('voxels', voxels)
	# voxels = np.array(voxels).reshape(-1)
	# with open(os.path.join('./data/iso_mesh/vor_000_bpp0.0983_iso8_128_400scan_normal.bin'), 'wb') as f: #x3
	# 	f.write(struct.pack('<%df' % len(voxels), *voxels)) 
	# vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
	# print(np.min(faces), np.max(faces))
	# vertices = np.array(mesh.vertices).reshape(-1)
	# faces = np.array(mesh.faces).reshape(-1)
	# print(vertices[-50:], len(vertices))
	# with open(os.path.join('./data/iso_mesh/vor_000_bpp0.0983_iso8_128_100scan_vertices.bin'), 'wb') as f: #x3
	# 	f.write(struct.pack('<%df' % len(vertices), *vertices)) 
	# with open(os.path.join('./data/iso_mesh/vor_000_bpp0.0983_iso8_128_100scan_faces.bin'), 'wb') as f: #x3
	# 	f.write(struct.pack('<%df' % len(faces), *faces)) 
	
	# mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
	# mesh.show()

	# query_points = np.mgrid[0:128, 0:128, 0:128].reshape(3, -1).T
	# voxels = mesh_to_sdf(mesh, query_points, surface_point_method='scan', sign_method='normal', bounding_radius=None, scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11)
	# voxels = np.array(voxels).reshape(-1)
	# with open(os.path.join('./data/iso_mesh/vor_000_bpp0.0983_iso8_mesh2sdf_128.bin'), 'wb') as f:
	# 	f.write(struct.pack('<%df' % len(voxels), *voxels)) 

	# points, sdf = sample_sdf_near_surface(mesh, number_of_points=250000)
	# print(np.array(points).shape)
	# print(np.array(sdf), np.array(sdf).shape)
	# colors = np.zeros(points.shape)
	# colors[sdf < 0, 2] = 1
	# colors[sdf > 0, 0] = 1
	# cloud = pyrender.Mesh.from_points(points, colors=colors)
	# scene = pyrender.Scene()
	# scene.add(cloud)
	# viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=20)

def minmax_range():
    data_file_path_txt = '/users/PAS0027/shen1250/Project/SciVis_Autoencoder/data/CL_isabel_trainingdata_P_txt/P_ts05_5500.txt'
    fh = open(data_file_path_txt, 'r')
    data_files = []
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        data_files.append(line)
    fh.close()
    
    min_ = 10000
    max_ = -10000
    
    for f in data_files:
        d = np.fromfile(f, '<f4').copy()
        if max_ < np.max(d):
            max_ = np.max(d)
        if min_ > np.min(d):
            min_ = np.min(d)
    print(min_, max_) #-4696.609 2102.8174


if __name__ == '__main__':
    # minmax_range()
	plot_mesh()


