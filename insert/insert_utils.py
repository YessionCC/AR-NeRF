import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2, sys, os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from tonemapping import *

#torch.set_default_tensor_type('torch.cuda.FloatTensor')

# vec: x, 3
def normalize_eps(vec, eps = 1e-6):
  return vec / (torch.norm(vec, dim = -1, keepdim=True)+eps)
def normalize(vec):
  return vec / torch.norm(vec, dim = -1, keepdim=True)
def normalize_np(vec):
  return vec / np.linalg.norm(vec, axis = -1, keepdims=True)

'''
im: w,h,3(rgb) or w,h(grey) GPU torch
'''
def show_im(im, inGPU = True):
  plt.figure()
  if inGPU:
    plt.imshow(im.cpu().numpy())
  else:
    plt.imshow(im)
  plt.show()

def show_im_cv(im, title = 'render'):
  cv2.imshow(title, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
  cv2.waitKey(1)

def write2ply(rgbs, pts, save_path):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(pts)
  pcd.colors = o3d.utility.Vector3dVector(rgbs)
  print('write ply file...')
  o3d.io.write_point_cloud(save_path, pcd, write_ascii=True)
  print('point cloud generate complete')

'''
pts: b,h,w,3
'''
def pts2normal(pts):
  b, w, h = pts.shape[:-1]
  pts = torch.Tensor(pts)
  dy = pts[:,:-1,...] - pts[:,1:,...]
  dy = torch.concat([dy[:,:1,...],dy], 1)
  dx = pts[...,:-1,:] - pts[...,1:,:]
  dx = torch.concat([dx[...,:1,:],dx], 2)
  normal = torch.cross(dy, dx, -1)
  return normalize(normal)

def get_sphere_rays(probe_num, ray_num):
  rds = torch.rand((2, probe_num, ray_num))
  rds[0] = 1.0 - 2.0*rds[0] #costheta
  rds[1] = 2.0*torch.pi*rds[1] #phi
  sintheta = torch.sqrt(1.0 - torch.clamp(rds[0]*rds[0], 0.0, 1.0))
  x = sintheta * torch.cos(rds[1])
  y = sintheta * torch.sin(rds[1])
  z = rds[0]
  dirs = torch.stack([x, y, z], dim = -1) # probe_num, ray_num, 3
  return dirs

def get_sphere_rays_np(probe_num, ray_num):
  rds = np.random.rand(2, probe_num, ray_num)
  rds[0] = 1.0 - 2.0*rds[0] #costheta
  rds[1] = 2.0*np.pi*rds[1] #phi
  sintheta = np.sqrt(1.0 - np.clip(rds[0]*rds[0], 0.0, 1.0))
  x = sintheta * np.cos(rds[1])
  y = sintheta * np.sin(rds[1])
  z = rds[0]
  dirs = np.stack([x, y, z], axis = -1) # probe_num, ray_num, 3
  return dirs

def get_cubemap_rays(probe_num, resolution, keep_raw_dim = False):
  x = torch.linspace(0,1,resolution)*2 -1
  X, Y = torch.meshgrid(x, x)
  X = X.unsqueeze(-1)
  Y = Y.unsqueeze(-1)
  oss = torch.ones((int(resolution), int(resolution), 1))
  front = torch.concat([X, Y, oss], dim = -1) # +z
  back = torch.concat([X, Y, -oss], dim = -1) # -z
  left = torch.concat([oss, X, Y], dim = -1) # +x
  rght = torch.concat([-oss, X, Y], dim = -1) # -x
  up = torch.concat([X, oss, Y], dim = -1) # +y
  down = torch.concat([X, -oss, Y], dim = -1) # -y
  dirs = torch.stack([front, back, left, rght, up, down], dim = 0) #6rr3
  dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
  if keep_raw_dim:
    return dirs
  dirs = dirs.reshape(-1, 3).unsqueeze(0)
  return dirs.expand(probe_num, *dirs.shape[1:])

SH_functions = [
  # order 2
  lambda x: 0.4886025119*x[..., 1],
  lambda x: 0.4886025119*x[..., 2],
  lambda x: 0.4886025119*x[..., 0],
  # order 3
  lambda x: 1.0925484306*x[..., 0]*x[..., 1],
  lambda x: 1.0925484306*x[..., 1]*x[..., 2],
  lambda x: 0.3153915653*(3.0*x[...,2]**2 - 1),
  lambda x: 1.0925484306*x[..., 0]*x[..., 2],
  lambda x: 0.5462742153*(x[..., 0]**2 - x[..., 1]**2)
]
'''
  # order 4
  lambda x: 0.5900435899*x[..., 1]*(3*x[...,0]**2 - x[...,1]**2),
  lambda x: 2.8906114426*x[..., 0]*x[..., 1]*x[..., 2],
  lambda x: 0.4570457995*x[..., 1]*(5*x[...,2]**2 - 1),
  lambda x: 0.3731763326*(5*x[..., 2]**3 - 3*x[..., 2]),
  lambda x: 0.4570457995*x[..., 0]*(5*x[...,2]**2 - 1),
  lambda x: 1.4453057213*x[..., 2]*(x[...,0]**2 - x[...,1]**2),
  lambda x: 0.5900435899*x[..., 0]*(x[...,0]**2 - 3*x[...,1]**2)
]
'''
# order 1
SH_functions_np = [lambda x: 0.2820947918*np.ones(x.shape[:-1])]+SH_functions
SH_functions_torch = [lambda x: 0.2820947918*torch.ones(x.shape[:-1])]+SH_functions

'''
rays_d, rays_rgb: probe_num, per_ray, 3
'''
def get_SH_coeff(rays_d, rays_rgb):
  shs = torch.stack([SH_function(rays_d) for SH_function in SH_functions_torch], dim = -1)
  shs_res = torch.matmul(shs.unsqueeze(-1), rays_rgb.unsqueeze(-2)) # 9,1 * 1，3 = 9, 3
  shs_res = torch.sum(shs_res, dim = 1)*4*torch.pi / rays_d.shape[1] # probe_num, 9, 3
  return shs_res

'''
shec: 9, 3
dirs: x,3
'''
def get_SH_val(shec, dirs, clamp_postive = False):
  dir_shs = torch.stack([SH_function(dirs) for SH_function in SH_functions_torch], dim = -1)#x,9
  vals = torch.matmul(dir_shs.unsqueeze(-2), shec).squeeze(-2)
  if clamp_postive:
    vals = F.relu(vals)
  return vals #x, 3

'''
  shec1, shec2: x,9
  triple product two SH, only return Y0
'''
def SH_product0(shec1, shec2):
  return 0.2821*torch.sum(shec1*shec2, dim = -1, keepdim=True)

# shec: x,9,3
def get_SH_main_direction(shec):
  dirc_R = torch.stack([shec[:,3, 0], shec[:,1,0], shec[:,2,0]], dim=-1)
  dirc_G = torch.stack([shec[:,3, 1], shec[:,1,1], shec[:,2,1]], dim=-1)
  dirc_B = torch.stack([shec[:,3, 2], shec[:,1,2], shec[:,2,2]], dim=-1)
  dirc = dirc_R*0.3 + dirc_G * 0.59 + dirc_B * 0.11 # x,3
  return normalize(dirc)

def get_cubemap_main_direction(cubemap, ray_dir):
  radiance = cubemap[:,0]*0.3+cubemap[:,1]*0.59+cubemap[:,2]*0.11
  max_dir_id = torch.argmax(radiance.flatten())
  return ray_dir[:,max_dir_id, :]

# ray_dir, ray_rgb: x,3
# rot_mat: 3*3
def rotate_SH_by_recalc(ray_dir, ray_rgb, rot_mat):
  ray_dir = (rot_mat @ ray_dir.T).T
  return get_SH_coeff(ray_dir[None,...], ray_rgb[None,...])

'''
sh_coeff: [9,3]: gpu torch
resolution: int
'''
def visualize_SH(sh_coeff, resolution, hdr = False, use_cv = False):
  dirs = get_cubemap_rays(1, resolution, True)
  rgbs = get_SH_val(sh_coeff, dirs)

  painter = torch.ones((resolution*3, resolution*4, 3))
  painter[resolution:resolution*2, resolution:resolution*2,:] = rgbs[0].transpose(0,1)
  painter[resolution:resolution*2, resolution*3:resolution*4,:] = torch.flip(rgbs[1].transpose(0,1), [1])
  painter[resolution:resolution*2, resolution*2:resolution*3,:] = torch.flip(rgbs[2],[1])
  painter[resolution:resolution*2, 0:resolution,:] = rgbs[3]
  painter[resolution*2:resolution*3, resolution:resolution*2,:] = torch.flip(rgbs[4].transpose(0,1), [0]) 
  painter[0:resolution, resolution:resolution*2,:] = rgbs[5].transpose(0,1)

  if hdr:
    painter = tonemapping_simple_torch(painter)
  
  #painter = (painter.cpu().numpy()*255).astype('uint8')
  if use_cv:
    show_im_cv(painter.cpu().numpy(), 'SH view')
  else:
    show_im(painter)

# sh_coeff: 9,3
def SH2Envmap_forDraw(sh_coeff, H, W, upper_hemi=False):
  if upper_hemi:
    phi, theta = torch.meshgrid([torch.linspace(0., np.pi/2., H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])
  else:
    phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])

  viewdirs = torch.stack([
    torch.cos(theta) * torch.sin(phi), 
    torch.cos(phi), 
    torch.sin(theta) * torch.sin(phi)], dim=-1)    # [H, W, 3]
  viewdirs = viewdirs.reshape(-1, 3)
  rgb = get_SH_val(sh_coeff, viewdirs)
  envmap = rgb.reshape((H, W, 3))
  return envmap # H,W,3

'''
ray_rgb: 2, ray_num, 3:   0:dir, 1:rgb, gpu torch
'''
sel_mask = torch.tensor([[1,2],[0,2],[0,1]], dtype=torch.int64)
def visualize_env(ray_rgb, resolution, hdr = False, use_cv = False, cv_name = 'env view'):
  rgbs = torch.zeros((6, resolution, resolution, 3), dtype=torch.float)
  ray_d, rgb = ray_rgb
  max_ax, max_id= torch.max(torch.abs(ray_d), -1)
  ray_d = ray_d / max_ax.unsqueeze(-1)
  def get_pos(select_axis):
    mask = sel_mask[select_axis]
    xx = ray_d[max_id==select_axis,:]
    rx = rgb[max_id==select_axis,:]
    posx = xx[xx[:,select_axis]>0,:]
    posx = torch.clamp((posx[:, mask]*0.5+0.5)*resolution, 0, resolution-1).long()
    posrgb = rx[xx[:,select_axis]>0,:]
    negx = xx[xx[:,select_axis]<0,:]
    negx = torch.clamp((negx[:, mask]*0.5+0.5)*resolution, 0, resolution-1).long()
    negrgb = rx[xx[:,select_axis]<0,:]
    

    rgbs[select_axis*2, posx[:,0], posx[:,1], :] = posrgb
    rgbs[select_axis*2+1, negx[:,0], negx[:,1], :] = negrgb

  get_pos(0)
  get_pos(1)
  get_pos(2)

  painter = torch.ones((resolution*3, resolution*4, 3))
  painter[resolution:resolution*2, resolution:resolution*2,:] = rgbs[4].transpose(0,1)
  painter[resolution:resolution*2, resolution*3:resolution*4,:] = torch.flip(rgbs[5].transpose(0,1), [1])
  painter[resolution:resolution*2, resolution*2:resolution*3,:] = torch.flip(rgbs[0],[1])
  painter[resolution:resolution*2, 0:resolution,:] = rgbs[1]
  painter[resolution*2:resolution*3, resolution:resolution*2,:] = torch.flip(rgbs[2].transpose(0,1), [0]) 
  painter[0:resolution, resolution:resolution*2,:] = rgbs[3].transpose(0,1)

  if hdr:
    painter = tonemapping_simple_torch(painter)

  if use_cv:
    show_im_cv(painter.cpu().numpy(), cv_name)
  else:
    show_im(painter)

def visualize_env_rad(ray_rgb, resolution):
  rgbs = torch.zeros((6, resolution, resolution, 3), dtype=torch.float)
  ray_d, rgb = ray_rgb
  max_ax, max_id= torch.max(torch.abs(ray_d), -1)
  ray_d = ray_d / max_ax.unsqueeze(-1)
  def get_pos(select_axis):
    mask = sel_mask[select_axis]
    xx = ray_d[max_id==select_axis,:]
    rx = rgb[max_id==select_axis,:]
    posx = xx[xx[:,select_axis]>0,:]
    posx = torch.clamp((posx[:, mask]*0.5+0.5)*resolution, 0, resolution-1).long()
    posrgb = rx[xx[:,select_axis]>0,:]
    negx = xx[xx[:,select_axis]<0,:]
    negx = torch.clamp((negx[:, mask]*0.5+0.5)*resolution, 0, resolution-1).long()
    negrgb = rx[xx[:,select_axis]<0,:]
    

    rgbs[select_axis*2, posx[:,0], posx[:,1], :] = posrgb
    rgbs[select_axis*2+1, negx[:,0], negx[:,1], :] = negrgb

  get_pos(0)
  get_pos(1)
  get_pos(2)

  painter = torch.ones((resolution*3, resolution*4, 3))
  painter[resolution:resolution*2, resolution:resolution*2,:] = rgbs[4].transpose(0,1)
  painter[resolution:resolution*2, resolution*3:resolution*4,:] = torch.flip(rgbs[5].transpose(0,1), [1])
  painter[resolution:resolution*2, resolution*2:resolution*3,:] = torch.flip(rgbs[0],[1])
  painter[resolution:resolution*2, 0:resolution,:] = rgbs[1]
  painter[resolution*2:resolution*3, resolution:resolution*2,:] = torch.flip(rgbs[2].transpose(0,1), [0]) 
  painter[0:resolution, resolution:resolution*2,:] = rgbs[3].transpose(0,1)

  painter = torch.sum(painter, -1)
  show_im(painter)


class SH9_Triple_Product:
  def __init__(self):
    clebsch_3 = torch.load('./insert/data/clebsch_3.tar')
    self.cidx = clebsch_3['idx'].cuda()
    self.cval = clebsch_3['val'].cuda()

  '''
  shec1, shec2: x,9
  '''
  def SH9_product(self, shec1, shec2):
    shres = torch.zeros_like(shec1)
    for i in range(self.cidx.shape[0]):
      shres[..., self.cidx[i,2]] += \
        self.cval[i] * shec1[..., self.cidx[i,0]] * shec2[..., self.cidx[i,1]]
    return shres # x,9

  '''
  shec1, shec2: x,9,3
  '''
  def SH9_product_93(self, shec1, shec2):
    res = self.SH9_product(shec1.permute(0,2,1), shec2.permute(0,2,1))
    return res.permute(0,2,1)