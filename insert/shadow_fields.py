import torch
import numpy as np

from tqdm import tqdm

from insert_utils import *
from render_utils import *
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def gen_sf_3d():
  range_i = 6
  step = 128
  oxs = torch.linspace(-range_i, range_i, step)
  oos = torch.meshgrid(oxs, oxs, oxs)
  oos = torch.stack(oos, -1)
  shs = []
  #oos = torch.Tensor([[0.5,0,0], [0,0.5,0], [0,0,0.5]])
  for ox in tqdm(oos.reshape(-1,3)):
    #print(ox)
    rays_d = get_sphere_rays(1, 8192*16)[0]
    #rays_d = get_cubemap_rays(len(oxs), 256)[0]
    visi = torch.zeros(rays_d.shape[0])
    ox2 = torch.sum(ox**2, -1)
    oxd = torch.sum(ox*rays_d, -1)
    dist = torch.min(torch.Tensor([1]), ox2)
    delta = oxd**2 - ox2 + dist
    visi[delta <= 0] = 1.0
    sqr_delta = torch.sqrt(delta) 
    t1 = -oxd+ sqr_delta
    t2 = -oxd - sqr_delta
    t = torch.logical_or(t1<0, t2<0)
    visi[t] = 1.0
    visi = visi[...,None].expand_as(rays_d)
    sh_coeff = get_SH_coeff(rays_d.unsqueeze(0), visi.unsqueeze(0))
    shs.append(sh_coeff)
    visualize_SH(sh_coeff[0], 48)
    visualize_env(torch.stack([rays_d, visi], 0), 256)
  shs = torch.concat(shs, 0)
  shs = shs.reshape(step, step, step, 9, 3)[...,0]
  torch.save(shs, './insert/data/sf.tar')
  return shs


def transform_sf_txt_to_torch(path_sh, save_path):
  info_dict = torch.Tensor(np.loadtxt(path_sh)).reshape(30,30,30,-1) #xyz 9/16
  info_dict = info_dict.permute(3,2,1,0).unsqueeze(0) #1,9/16,zyx
  torch.save(info_dict, save_path)

'''
model_pos: float3
model_r: float
model_sh9: 1, 9, 3
pts, pts_n: x,3 
return: x, 1
'''
def soft_shadow_map(sfer, model_pos, model_r, model_sh9, pts, rot_inv = None):
  m2pts = pts - model_pos.unsqueeze(0)
  #same_side = torch.sum(pts2m * pts_n, -1) > 0
  #same_side = torch.sum(pts2m * pts_n, -1) != 0
  #res = torch.ones(pts.shape[0])
  #pts = pts[same_side]
  #pts_n = pts_n[same_side]
  #m2pts = -pts2m[same_side]
  if rot_inv is not None:
    m2pts = (rot_inv @ m2pts.T).T

  pts_sh9 = sfer.fetch_sh(model_r, m2pts) # x,9
  psh = SH_product0(
    pts_sh9.unsqueeze(1).expand(pts.shape[0], 3, sfer.sh_coeff_num), # x, 3, 9
    model_sh9.permute(0,2,1) # 1, 3, 9
  )#.permute(0,2,1) # x, 9, 3
  #old_ir = SH9_irradiance(pts_n, model_sh9) # 1,9,3
  #new_ir = SH9_irradiance(pts_n, psh) # x,9,3

  old_ir = model_sh9[:,0,:] # 1,3
  #new_ir = psh[:,0,:] # x,3
  new_ir = psh[..., 0] # x,3
  
  res = torch.mean(torch.clamp(new_ir / old_ir, 0.0, 1.0), dim = -1)
  
  res = torch.pow(res, 10) # to augment shadow effect

  return res


class SimplifySF:
  def __init__(self, sh_coeff_num = 9):
    self.vol_range = 6
    self.sh_coeff_num = sh_coeff_num
    self.sf_vol = torch.load('./insert/data/sf.tar').cuda() # XYZ 9
    self.sf_vol = self.sf_vol.permute(3,2,1,0).unsqueeze(0) # 1,9,ZYX (DHW)

  '''
  scale: float
  pts: x, 3
  '''
  def fetch_sh(self, scale, pts):
    pts = pts / scale / self.vol_range
    pts = pts.reshape(1,1,1,-1,3)
    pt_sh9 = F.grid_sample(
      self.sf_vol, pts, 
      mode='bilinear', padding_mode='border', 
      align_corners=True
    )
    pt_sh9 = pt_sh9.permute(0,2,3,4,1).reshape(-1, self.sh_coeff_num) # x,9
    return pt_sh9

class ComplexSF:
  def __init__(self, sh_path, sh_coeff_num = 9):
    self.vol_range = 4
    self.sh_coeff_num = sh_coeff_num
    self.sf_vol = torch.load(sh_path).cuda() # XYZ 9

  '''
  scale: float
  pts: x, 3
  '''
  def fetch_sh(self, scale, pts):
    pts = pts / scale / self.vol_range
    pts = pts.reshape(1,1,1,-1,3)
    pt_sh9 = F.grid_sample(
      self.sf_vol, pts, 
      mode='bilinear', padding_mode='border', 
      align_corners=True
    )
    pt_sh9 = pt_sh9.permute(0,2,3,4,1).reshape(-1, self.sh_coeff_num) # x,9
    return pt_sh9

def __test(sh_coeff_num = 9):
  t = SimplifySF(sh_coeff_num)
  pts = torch.Tensor([0,1.5,0]).reshape(-1, 3)
  shs = t.fetch_sh(1, pts)[0]
  shs = shs.reshape(sh_coeff_num,1).expand(sh_coeff_num,3)
  visualize_SH(shs, 48)

def ___test(sh_coeff_num = 9):
  t = ComplexSF("dinosaur_sh.tar", sh_coeff_num)
  pts = torch.Tensor([0,0,-1.2]).reshape(-1, 3)
  shs = t.fetch_sh(1, pts)[0]
  shs = shs.reshape(sh_coeff_num,1).expand(sh_coeff_num,3)
  visualize_SH(shs, 48)

#gen_sf_3d()
#__test()
#___test(16)
#transform_sf_txt_to_torch("horse_sh.txt", "horse_sh.tar")