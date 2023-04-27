import torch
import torch.nn as nn
import torch.nn.functional as F
import os, cv2
import numpy as np
from tqdm import tqdm
import pyransac3d as pyrsc

from insert_utils import *

import matplotlib.pyplot as plt

torch.set_default_tensor_type('torch.cuda.FloatTensor')


class GlobalLightEstimator:
  def __init__(self, gen_path, pts_use = int(2e6), write_ply = False):
    self.calc_complete = False
    self.write_ply = write_ply
    self.save_path = os.path.join(gen_path, 'plane.npy')
    if os.path.exists(self.save_path):
      plane_infos = np.load(self.save_path, allow_pickle=True).item()
      self.t_rgbs = plane_infos['rgbs'].reshape(-1, 3)
      self.t_pts = plane_infos['spts'].reshape(-1, 3)
      self.t_normal = plane_infos['normals'].reshape(-1, 3)
      if 'rgb_shs' in plane_infos:
        self.t_rgb_shs = plane_infos['rgb_shs']
        self.t_opc_shs = plane_infos['opacity_shs']
      print('Find plane infos, {} points will be used in training'.
        format(self.t_pts.shape[0]))
      self.calc_complete = True
    else:
      surface_infos = np.load(
        os.path.join(gen_path, 'surface.npy'), allow_pickle=True).item()
      s_rgbs = surface_infos['rgbs'].reshape(-1, 3)
      s_pts = surface_infos['spts'].reshape(-1, 3)
      s_normals = surface_infos['normals'].reshape(-1, 3)
      shuffle_idx = np.random.permutation(s_pts.shape[0])
      s_rgbs = s_rgbs[shuffle_idx]
      s_pts = s_pts[shuffle_idx]
      s_normals = s_normals[shuffle_idx]
      self.s_rgbs = s_rgbs[:pts_use]
      self.s_pts = s_pts[:pts_use]
      self.s_normals = s_normals[:pts_use]
      self.pts_num = pts_use

      self.t_rgbs = []
      self.t_pts = []
      self.t_normal = []

  def detect_planar_patch(self, min_pts_in_plane = 1e5):
    pt_c = self.s_pts
    rgb_c = self.s_rgbs
    norm_c = self.s_normals
    if self.write_ply:
      self.rgb_msk = []
    while True:
      plane1 = pyrsc.Plane()
      best_eq, best_inliers = plane1.fit(pt_c, 0.02)
      if best_inliers.shape[0] < min_pts_in_plane: break

      normal = np.array(best_eq[:3]).reshape(1,3)
      raw_mean_normal = np.mean(norm_c[best_inliers], 0, keepdims=True)
      rdcn = np.sum(normal*raw_mean_normal)
      if rdcn < 0: # only use raw normal correct direction
        normal = -normal
      normal = normalize_np(normal)
      print('Find plane, normal: ', normal)
      self.t_rgbs.append(rgb_c[best_inliers])
      self.t_pts.append(pt_c[best_inliers])
      self.t_normal.append(normal.repeat(len(best_inliers), axis = 0))
      if self.write_ply:
        col_msk = np.random.random((1,3)).repeat(len(best_inliers), axis = 0)
        self.rgb_msk.append(col_msk)

      mask = np.ones(pt_c.shape[0], dtype=np.bool8)
      mask[best_inliers] = False
      pt_c = pt_c[mask]
      rgb_c = rgb_c[mask]
      norm_c = norm_c[mask]
    
    self.t_rgbs = np.concatenate(self.t_rgbs, 0)
    self.t_pts = np.concatenate(self.t_pts, 0)
    self.t_normal = np.concatenate(self.t_normal, 0)

  def save_results(self, nerf_model = None):
    pts_num = self.t_pts.shape[0]
    save_dict = {
      'spts': self.t_pts,
      'rgbs': self.t_rgbs,
      'normals': self.t_normal
    }
    if nerf_model is not None:
      batch = 4096
      rgb_shs = []
      opacity_shs = []
      print('Precompute probes ...')
      for i in tqdm(range(0, pts_num, batch)):
        ed = min(i+batch, pts_num)
        rgb_sh_coeff, opacity_sh_coeff = \
          nerf_model.generate_SH_probes_for_precompute(
            torch.Tensor(self.t_pts[i:ed]) + torch.Tensor(self.t_normal[i:ed])*0.01
          )
        rgb_shs.append(rgb_sh_coeff.cpu().numpy())
        opacity_shs.append(opacity_sh_coeff.cpu().numpy())
      self.t_rgb_shs = np.concatenate(rgb_shs, axis=0) # x,9,3
      self.t_opc_shs = np.concatenate(opacity_shs, axis=0) # x,9,1
      save_dict.update({
        'rgb_shs': self.t_rgb_shs,
        'opacity_shs': self.t_opc_shs
      })
    
    print('{} points will be used in training'.format(pts_num))
    np.save(self.save_path, save_dict, allow_pickle=True)

    if self.write_ply:
      self.rgb_msk = np.concatenate(self.rgb_msk, 0)
      write2ply(self.s_rgbs, self.s_pts, './scene_sample.ply')
      write2ply(self.rgb_msk, self.t_pts, './scene_plane.ply')



def __test():
  gle = GlobalLightEstimator('./insert/generate/Test')
  gle.detect_planar_patch(write_ply=False)

if __name__ == '__main__':
  __test()