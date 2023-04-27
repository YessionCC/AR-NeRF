import torch
import numpy as np

from tqdm import tqdm

from insert_utils import *
from render_utils import *
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class SGShadow:
  def __init__(self, pca_path, grid_size = 20, 
    ncomponents = 32, vol_range = 4, envH = 128, envW = 128):
    self.delta_angle_decay_fac = 0.4
    self.vol_range = vol_range
    self.raw_h_angle = torch.atan(torch.Tensor([1.0/vol_range]))
    self.ncomponents = ncomponents
    self.envH = envH
    self.envW = envW
    self.fh_tab = torch.Tensor(
      np.load('./insert/data/fh_pretab.npy')) #2048(lbd)，1024(theta_d)
    #show_im(self.fh_tab)
    self.fh_tab = self.fh_tab[None, None, ...] # 1,1,2048,1024
    data = torch.load(pca_path)
    self.coeff_volume = data['coeff'].reshape(
      grid_size, grid_size, grid_size, ncomponents #20,20,20,32
      ).permute(3,2,1,0).unsqueeze(0).cuda() # 1,32,20,20,20 # 1,C,D,HW
    self.components = data['component'].cuda() # 32,128,128
    self.mean = data['mean'].cuda() # 1,128,128


  def light_axis_to_cood(self, lSGs):
    phi = torch.arccos(lSGs[:, 1])
    theta = torch.arctan2(lSGs[:,2], lSGs[:,0]) 
    phi_n = phi / torch.pi * 2 - 1
    theta_n = theta / torch.pi
    #lAxisPos2D = torch.stack([phi_n, theta_n], -1) # lx,2
    '''
    !!!! grid_sample 若是二维, x对应W方向, y对应H方向!!!!!
          若是三维, z对应D方向
          无论是哪一轴，-1对应序号0, 1对应最后一个
      使用grid_sample一定要检查输出是否正确!!!!!!
    '''
    lAxisPos2D = torch.stack([theta_n, phi_n], -1) # lx,2
    lAxisPos2D = lAxisPos2D.reshape(1,1,-1,2)

    components_s = F.grid_sample(self.components.unsqueeze(0), lAxisPos2D, padding_mode="border") #1,32,1,lx
    self.components_s = components_s.permute(0,2,3,1)[0,0,...] # lx,32

    mean_s = F.grid_sample(self.mean.unsqueeze(0), lAxisPos2D, padding_mode="border") # 1,1,1,lx
    self.mean_s = mean_s[0,0,0,:].unsqueeze(0) # 1,lx

  def calc_inte_L_V(self, ssdf, lSGs):
    ssdf = ssdf / (torch.pi/2)
    # logspace: -1~4 -> -1~1
    lambdas = (torch.log10(torch.abs(lSGs[:,3]+1e-6)) - 1.5)/2.5
    lambdas = lambdas.unsqueeze(0).expand_as(ssdf) #px,lx

    ssdf_lbds = torch.stack([ssdf, lambdas], 0) # 2,px,lx
    ssdf_lbds = ssdf_lbds.permute(1,2,0).unsqueeze(0) # 1,px,lx,2

    fhs = F.grid_sample(self.fh_tab, ssdf_lbds, padding_mode="border")# 1,1,px,lx
    lcols = lSGs[:,-3:] # lx,3
    cols = fhs[0,0,...] @ lcols # px,3
    return cols

  def calc_inte_L(self, lSGs): # hemisphere integral
    expTerm = 1. - torch.exp(-1. * lSGs[:,3:4])
    cols = 2 * torch.pi * (lSGs[:,-3:] / lSGs[:,3:4]) * expTerm # lx,3
    cols = torch.sum(cols, 0, keepdim=True) # 1,3
    return cols

  '''
  scale: float
  pts: x, 3
  '''
  def fetch_ssdf(self, scale, pts):
    pts = pts / scale / self.vol_range
    # for pts out of range, normalize it
    dis = torch.norm(pts, dim = -1, keepdim=True).clip(min = 1)
    pts = pts / dis

    cur_h_angle = torch.atan(1.0/(dis*self.vol_range))
    delta_h_angle = self.raw_h_angle - cur_h_angle # px,1
    delta_h_angle *= self.delta_angle_decay_fac

    pts = pts.reshape(1,1,1,-1,3)
    pca32 = F.grid_sample(
      self.coeff_volume, pts, 
      mode='bilinear', padding_mode='border', 
      align_corners=True
    )
    pca32 = pca32.permute(0,2,3,4,1).reshape(-1, self.ncomponents) # px,32

    ssdf = pca32 @ self.components_s.T + self.mean_s # px, lx

    ssdf += delta_h_angle

    return ssdf

  def calc_shadow_factor(self, scale, pts, model_pos, lSGs, rot_inv = None):
    m2pts = pts - model_pos.unsqueeze(0)
    if rot_inv is not None:
      m2pts = (rot_inv @ m2pts.T).T

    self.light_axis_to_cood(lSGs)
    ssdf = torch.clip(self.fetch_ssdf(scale, m2pts), -torch.pi/2, torch.pi/2)
    inte_L_V = self.calc_inte_L_V(ssdf, lSGs)#px,3
    inte_L = self.calc_inte_L(lSGs) #1,3
    factor = torch.abs(inte_L_V / inte_L).clip(0, 1)
    factor = 0.2989 * factor[:,0] + 0.5870 * factor[:,1] + 0.1140*factor[:,2]
    factor = torch.pow(factor, 8) ############################## WARNING HACK
    return factor # px
    #return ssdf[:,0].clip(0,1)

  def calc_self_shadow_light_dacay(
    self, scale, pts, model_pos, lSGs, rot_inv = None):

    m2pts = pts - model_pos.unsqueeze(0)
    if rot_inv is not None:
      m2pts = (rot_inv @ m2pts.T).T
      lSGs_rot = lSGs.clone()
      lSGs_rot[:, :3] = (rot_inv @ lSGs_rot[:, :3].T).T
      self.light_axis_to_cood(lSGs_rot)
    else:
      self.light_axis_to_cood(lSGs)
    ssdf = torch.clip(self.fetch_ssdf(scale, m2pts), -torch.pi/2, torch.pi/2)

    ssdf = ssdf / (torch.pi/2)
    # logspace: -1~4 -> -1~1
    lambdas = (torch.log10(torch.abs(lSGs[:,3]+1e-6)) - 1.5)/2.5
    lambdas = lambdas.unsqueeze(0).expand_as(ssdf) #px,lx

    ssdf_lbds = torch.stack([ssdf, lambdas], 0) # 2,px,lx
    ssdf_lbds = ssdf_lbds.permute(1,2,0).unsqueeze(0) # 1,px,lx,2

    fhs = F.grid_sample(self.fh_tab, ssdf_lbds, #mode='nearest',
      padding_mode="border")# 1,1,px,lx
    fhs = fhs[0,0,...] # px,lx

    expTerm = 1. - torch.exp(-1. * lSGs[:,3:4])
    fh_ns = 2 * torch.pi / lSGs[:,3:4] * expTerm # lx,1

    decay = torch.abs(fhs / fh_ns.T).clip(0,1).unsqueeze(-1) # px,lx,1
    decay = torch.pow(decay, 0.2) #

    lSGs_dec = lSGs[:,-3:].unsqueeze(0) # 1,lx,3
    lSGs_dec = lSGs_dec * decay # px,lx,3
    t = lSGs[:,:4].unsqueeze(0).expand(decay.shape[0], decay.shape[1], 4)
    res = torch.concat([t, lSGs_dec], -1) # px,lx,7
    return res#, ssdf


  # wrong. deprecated
  def __calc_inte_L_V(self, ssdf, lSGs):
    lambdas = lSGs[:,3].unsqueeze(0) #1,lx
    fh_n = 2*torch.pi / lambdas * (1 - torch.exp(-lambdas)) # 1,lx
    k_lbd = 0.204*lambdas**3 - 0.892*lambdas**2 + 2.995*lambdas + 0.067 # 1,lx
    a = 1.05
    #fh_x = a / (1+k_lbd*torch.exp(ssdf)) + (1-a)/2 # px,lx
    fh_x = a / (1+torch.exp(-k_lbd*ssdf)) + (1-a)/2 # px,lx
    fh_x = torch.clip(fh_x, 0, 1)
    fh = fh_n * fh_x # px,lx
    lcols = lSGs[:,-3:] # lx,3
    cols = fh @ lcols # px,3
    return cols

  # wrong. deprecated
  def __calc_inte_L_V_div_inte_L(self, ssdf, lSGs):
    lambdas = lSGs[:,3].unsqueeze(0) #1,lx
    k_lbd = 0.204*lambdas**3 - 0.892*lambdas**2 + 2.995*lambdas + 0.067 # 1,lx
    a = 1.05
    #fh_x = a / (1+k_lbd*torch.exp(ssdf)) + (1-a)/2 # px,lx
    fh_x = a / (1+torch.exp(-k_lbd*ssdf)) + (1-a)/2 # px,lx
    fh_x = torch.clip(fh_x, 0, 1)
    lcols = lSGs[:,-3:] # lx,3
    cols = fh_x @ lcols # px,3
    return cols

  '''
  pt: 3
  '''
  # only for debug
  def _fetch_ssdf_im(self, pt):
    pt = pt.reshape(1,1,1,-1,3)
    pca32 = F.grid_sample(
      self.coeff_volume, pt, 
      mode='bilinear', padding_mode='border', 
      align_corners=True
    )
    pca32 = pca32.permute(0,2,3,4,1).reshape(-1, self.ncomponents) # 1,32
    cps = pca32 @ self.components.reshape(self.ncomponents, -1)
    cps = cps.reshape(self.envH, self.envW)
    ssdf_im = self.mean[0] + cps #128,128
    return ssdf_im
  # only for debug
  def _deubug(self, pt, lgtDir, scale, model_pos, rot_inv = None):
    m2pt = pt - model_pos.unsqueeze(0)
    if rot_inv is not None:
      m2pt = (rot_inv @ m2pt.reshape(3,1)).flatten()
      lgtDir = (rot_inv @ lgtDir.T).T
    m2pt = m2pt / scale / self.vol_range
    ssdf_im = self._fetch_ssdf_im(m2pt)

    phi = torch.arccos(lgtDir[1])
    theta = torch.arctan2(lgtDir[2], lgtDir[0]) 
    phi_n = phi / torch.pi * 2 - 1 # -1~1
    theta_n = theta / torch.pi

    phi_n = int((0.5*phi_n + 0.5)*self.envH)
    theta_n = int((0.5*theta_n + 0.5)*self.envW)
    ssdf_im[phi_n, theta_n] = -ssdf_im[phi_n, theta_n]
    show_im(ssdf_im)


  

def __test():
  sgs = SGShadow('/home/lofr/Projects/Render/objViewer/bin/sg/results/Buddha.tar')
  pt = torch.Tensor([-0.3400,  0.0685, -0.2110])
  pt = torch.Tensor([-0.02, 0, 0])
  im = sgs.fetch_ssdf_im(pt)
  pt = torch.Tensor([0.7074, -0.2469, -0.6623])
  phi = torch.arccos(pt[1])
  theta = torch.arctan2(pt[2], pt[0]) 
  phi_n = phi / torch.pi * 2 - 1 # -1~1
  theta_n = theta / torch.pi

  phi_n = int((0.5*phi_n + 0.5)*128)
  theta_n = int((0.5*theta_n + 0.5)*128)
  im[phi_n, theta_n] *= 2
  show_im(im)

#__test()

def inte(lbd, theta_d):
  from scipy import integrate
  def inte_func(zeta, delta):
    return np.exp(lbd*(np.sin(zeta)*np.sin(delta) - 1))*np.sin(zeta)

  res = integrate.dblquad(inte_func, np.pi / 2 - theta_d, np.pi, 0, np.pi)
  return res[0]

def calc_inte_L(lbd): # hemisphere integral
  expTerm = 1. - np.exp(-1. * lbd)
  res = 2 * np.pi / lbd * expTerm # lx,3
  return res

def div1(lbd, theta_d):
  return inte(lbd, theta_d) / calc_inte_L(lbd)

def div2(lbd, theta_d):
  k_lbd = 0.204*lbd**3 - 0.892*lbd**2 + 2.995*lbd + 0.067 # 1,lx
  a = 1.05
  #fh_x = a / (1+k_lbd*torch.exp(ssdf)) + (1-a)/2 # px,lx
  fh_x = a / (1+np.exp(-k_lbd*theta_d)) + (1-a)/2 # px,lx
  fh_x = np.clip(fh_x, 0, 1)
  return fh_x

def pretabulate(theta_num = 1024, lbd_num = 2048):
  theta_ds = np.linspace(-np.pi/2, np.pi/2, theta_num)
  lbds = np.linspace(-1, 4, lbd_num)
  lbds = 10**lbds
  res = np.ones((lbd_num, theta_num), np.float32)
  for i,lbd in enumerate(lbds):
    print(i)
    for j,theta_d in enumerate(theta_ds):
      tr = inte(lbd, theta_d)
      res[i, j] = tr
  np.save('fh_pretab.npy', res)


#pretabulate()