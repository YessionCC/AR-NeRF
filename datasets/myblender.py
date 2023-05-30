import torch
import numpy as np
import os
import glob
from tqdm import tqdm

from .ray_utils import *
from .color_utils import read_image
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary

from .base import BaseDataset


class MyBlender(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self):
        self.K = torch.FloatTensor(np.loadtxt(os.path.join(self.root_dir, 'int.txt')))
        W = int(self.K[0,2])*2
        H = int(self.K[1,2])*2
        self.img_wh = (W, H)
        self.directions = get_ray_directions(H, W, self.K)

    def read_meta(self, split, **kwargs):
        exts = np.load(os.path.join(self.root_dir, 'exts.npy'))
        self.poses = []
        for ext in exts:
            ext = np.concatenate([ext, np.array([0,0,0,1]).reshape(1,4)], 0)
            self.poses.append(np.linalg.inv(ext))
        self.poses = np.stack(self.poses, 0)[:,:3,:]

        pose_radius_scale = 1.0

        scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()
        scale *= pose_radius_scale
        self.poses[..., 3] /= scale
        
        # self.blender_trans the NeRF coordinate to blender( raw coodrinate )
        self.blender_trans = np.eye(4)
        self.blender_scale = scale

        img_dir = os.path.join(self.root_dir, 'img')
        img_paths = [os.path.join(img_dir, im) for im in sorted(os.listdir(img_dir))]

        if len(img_paths) < self.poses.shape[0]:
            print('warning: use less img')
            self.poses = self.poses[:len(img_paths)]
        elif len(img_paths) > self.poses.shape[0]:
            print('error: incomplete pose')
        

        self.rays = []
        if split == 'test_traj': # use precomputed test poses
            self.poses = create_spheric_poses(1.2, self.poses[:, 1, 3].mean())
            self.poses = torch.FloatTensor(self.poses)
            return

        # use every 8th image as test set
        if split=='train':
            img_paths = [x for i, x in enumerate(img_paths) if i%8!=0]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i%8!=0])
        elif split=='test':
            img_paths = [x for i, x in enumerate(img_paths) if i%8==0]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i%8==0])
            

        print(f'Loading {len(img_paths)} {split} images ...')
        for img_path in tqdm(img_paths):
            buf = [] # buffer for ray attributes: rgb, etc

            img = read_image(img_path, self.img_wh, blend_a=False, exr_file=True)
            img = torch.FloatTensor(img)
            buf += [img]
            self.rays += [torch.cat(buf, 1)]

        self.rays = torch.stack(self.rays) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)