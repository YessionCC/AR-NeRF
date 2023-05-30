import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import gaussian_blur

EPS = 1e-6

def pos_dot(vec1, vec2):
  return F.relu(torch.sum(vec1*vec2, dim = -1, keepdim=True)) # x,1

def pos_dot_eps(vec1, vec2):
  return torch.clip(torch.sum(vec1*vec2, dim = -1, keepdim=True), min=EPS) # x,1

'''
normals: x,3
shec: x,9,3
'''
def SH9_irradiance(normals, shec, allow_neg = False):
  sh_ir1 = 0.42904276540489171563379376569857
  sh_ir2 = 0.51166335397324424423977581244463
  sh_ir3 = 0.24770795610037568833406429782001
  sh_ir4 = 0.88622692545275801364908374167057

  x = normals[:,0].unsqueeze(-1)
  y = normals[:,1].unsqueeze(-1)
  z = normals[:,2].unsqueeze(-1)
  irradiance = sh_ir1*(x**2 - y**2)*shec[:,8,:] \
    + (sh_ir3*(3.0*z**2 - 1))*shec[:,6,:] \
    + sh_ir4*shec[:,0,:] \
    + 2.0*sh_ir1*(shec[:,4,:]*x*y+shec[:,7,:]*x*z+shec[:,5,:]*y*z) \
    + 2.0*sh_ir2*(shec[:,3,:]*x+shec[:,1,:]*y+shec[:,2,:]*z)
  if allow_neg:
    return irradiance
  return F.relu(irradiance) # x,3

'''
rgbs: x, c, 3
rays_d: x, c, 3
normals: x, 3
'''
def irradiance_numerical(rgbs, rays_d, normals, allow_neg = False):
  dDotn = pos_dot(rays_d, normals.unsqueeze(1)) # x,c,1
  inte = torch.sum(dDotn * rgbs, dim=1) # x,3
  inte = inte * (4 * torch.pi / rays_d.shape[1])
  if allow_neg:
    return inte
  return F.relu(inte)


def get_F0(metal, albedo):
  return torch.ones_like(albedo)*0.04*(1.0 - metal) + albedo*metal


def fresnelSchlick(F0, HdotV):
  FF = F0 + (1.0 - F0)*torch.pow(1.0 - HdotV, 5)
  return FF # x,1

def fresnelSchlickRoughness(F0, NdotV, rough):
  FF = F0 + (torch.max((1.0 - rough).expand_as(F0), F0) - F0)*torch.pow(1.0 - NdotV, 5)
  return FF # x,1

def GeometrySchlickGGX(NdotV, roughness):
  r = (roughness + 1.0)
  k = (r*r) / 8.0
  return NdotV / (NdotV * (1.0 - k) + k)

def GeometryBlender(NdotV, roughness):
  a = roughness**2
  sqr_alpha_tan_n = a*(1.0/NdotV**2 - 1.0).clip(min=0.0)
  return 0.5*(torch.sqrt(1.0+sqr_alpha_tan_n)-1.0)

# map: H,W,3
# samples: x,2,   range(-1, 1)
# return: x,3
def tex2D(map, samples, reverseHW = False):
  if reverseHW:
    samples = torch.stack([samples[:,1], samples[:,0]], -1)
  map = map.permute(2, 0, 1).unsqueeze(0) # 1,3,H,W
  samples = samples.reshape(1,1,samples.shape[0],2) # 1,1,x,2
  res = F.grid_sample(map, samples, padding_mode="border") # 1,3,1,x
  res = res[0,:,0,:].T
  return res

# map: D,H,W,3
# samples: x,3 (2: HW, 1: D),  range(-1, 1)
# return x,3
def tex3D(map, samples, reverseHW = False):
  if reverseHW:
    samples = torch.stack([samples[:,1], samples[:,0], samples[:, 2]], -1)
  map = map.permute(3, 0, 1, 2).unsqueeze(0) # 1,3,D,H,W
  samples = samples.reshape(1,1,1,samples.shape[0],3) # 1,1,1,x,3
  res = F.grid_sample(map, samples, padding_mode="border") # 1,3,1,1,x
  res = res[0,:,0,0,:].T
  return res

# cubemap: 6,r,r,3
# r: int (odd)
def cubemap_blur(cubemap, r):
  res = []
  for i in range(6):
    res.append(gaussian_blur(cubemap[i].permute(2,0,1), (r,r)).permute(1,2,0))
  return torch.stack(res, 0)

# cubemap: 6,r,r,3
# r: int (odd)
def cubemap_blur_(cubemap, r):
  for i in range(6):
    cubemap[i] = gaussian_blur(cubemap[i].permute(2,0,1), (r,r)).permute(1,2,0)

# cubemap, ray_d: x, 3  
# resolution: float
# rough: x,1
# if blur_cm is false, rough should be None, and cubemap will not be blurred
sel_mask = torch.tensor([[1,2],[0,2],[0,1]], dtype=torch.int64)
sel_map = torch.tensor([2, 4, 0], dtype=torch.int64)
def cubemap_sample(cubemap, ray_d, resolution, rough = None, blur_cm = True):
  rgbs = torch.ones_like(ray_d)
  cubemap = cubemap.reshape(6, resolution, resolution, 3)

  if blur_cm:
    if rough is None: # just blur the cubemap to make 
      cubemap_blur_(cubemap, 3)
      cubemap_blur_(cubemap, 3)
    else:
      c0 = cubemap
      c1 = cubemap_blur(c0, 3)
      c2 = cubemap_blur(c1, 3)
      c3 = cubemap_blur(c2, 3)
      c4 = cubemap_blur(c3, 3)
      cube_rs = torch.stack([c0,c1,c2,c3,c4], 0) # 5,6,r,r,3

  max_ax, max_id= torch.max(torch.abs(ray_d), -1)
  ray_d = ray_d / max_ax.unsqueeze(-1)
  def get_pos(select_axis):
    mask = sel_mask[select_axis]
    axis_sel = max_id==select_axis
    xx = ray_d[axis_sel,:]
    rx = rgbs[axis_sel,:]

    pos_sel = xx[:,select_axis]>0
    neg_sel = xx[:,select_axis]<0
    posx = xx[pos_sel,:]
    negx = xx[neg_sel,:]

    if rough is None:
      #posx = torch.clamp((posx[:, mask]*0.5+0.5)*resolution, 0, resolution-1).long()
      #negx = torch.clamp((negx[:, mask]*0.5+0.5)*resolution, 0, resolution-1).long()
      #posrgb = cubemap[sel_map[select_axis], posx[:,0], posx[:,1], :]
      #negrgb = cubemap[sel_map[select_axis]+1, negx[:,0], negx[:,1], :]
      posrgb = tex2D(cubemap[sel_map[select_axis]], posx[:, mask], True)
      negrgb = tex2D(cubemap[sel_map[select_axis]+1], negx[:, mask], True)
    else:
      _rough = rough[axis_sel,:]*2 - 1 # from 0~1 -> -1~1
      posx = torch.concat([posx[:, mask], _rough[pos_sel,:]], -1)
      negx = torch.concat([negx[:, mask], _rough[neg_sel,:]], -1)
      posrgb = tex3D(cube_rs[:,sel_map[select_axis],...], posx, True)
      negrgb = tex3D(cube_rs[:,sel_map[select_axis]+1,...], negx, True)

    rx[pos_sel,:] = posrgb
    rx[neg_sel,:] = negrgb
    rgbs[axis_sel,:] = rx

  get_pos(0)
  get_pos(1)
  get_pos(2)
  return rgbs

# cubemap: x,3
# cm_resol: the size of cubemap
# H, W: output env_map size
env_vdirs = None
def cubemap2env_map(cubemap, cm_resol, H, W):
  global env_vdirs
  if env_vdirs is None:
    phi, theta = torch.meshgrid([
      torch.linspace(0., np.pi, H), 
      torch.linspace(-0.5*np.pi, 1.5*np.pi, W)
    ])

    env_vdirs = torch.stack([
      torch.cos(theta) * torch.sin(phi), 
      torch.cos(phi), 
      torch.sin(theta) * torch.sin(phi)
    ], dim=-1) # H, W, 3

  res = cubemap_sample(cubemap, env_vdirs.reshape(-1,3), cm_resol, None, False)
  res = res.reshape(H, W, 3)
  return res

def reflect_dir(normal, vdirs):
  return torch.sum(normal*vdirs, dim = -1, keepdim=True)*normal*2 - vdirs

def spec_shade(normal, vdirs, rough, kS, refl_probe):
  ray_refl = reflect_dir(normal, vdirs)
  refl_rgb = cubemap_sample(refl_probe, ray_refl, 32, rough)
  return kS * refl_rgb

def SH_glossy_shade(normal, vdirs, rough, model_brdf, embed_fn, sh9, F0):
  # 3+(3*2)*3 = 21, 21*2+1 = 43
  spec_sh9 = model_brdf(torch.concat([embed_fn(normal), embed_fn(vdirs), rough], -1)) #x, 18
  sh_num = sh9.shape[1]
  spec_sh9_1 = spec_sh9[:,:sh_num]
  spec_sh9_2 = spec_sh9[:,sh_num:]
  spec1 = sh9 * spec_sh9_1.unsqueeze(-1).expand(normal.shape[0], sh_num, 3)
  spec2 = sh9 * spec_sh9_2.unsqueeze(-1).expand(normal.shape[0], sh_num, 3)
  spec_col1 = torch.sum(spec1, 1)
  spec_col2 = torch.sum(spec2, 1)
  # specular not times kS, because the integration contain F
  return (F0*spec_col1 + spec_col2)


# if need specular shade, refl_probe must give
# if only need specular shader, refle_probe must give and only_spec = True
@torch.no_grad()
def SH_render_core(albedo, metal, rough, normal, vdirs, sh9, embed_fn, model_brdf, clamp01, refl_probe = None, only_spec = False):
  F0 = get_F0(metal, albedo)
  vdirs = -vdirs ## !!! vdir is camera to object, it should be inverse!

  NdotV = pos_dot(normal, vdirs)
  # Neural render at edge area is not stable, so detect edge and make the 
  # angle little less than 90
  edge_sel = [NdotV.flatten() < 8e-2]
  normal[edge_sel] += vdirs[edge_sel] / 10
  normal = normal/torch.norm(normal, dim=-1, keepdim=True)

  kS = fresnelSchlickRoughness(F0, NdotV, rough) ###
  kD = (1.0 - kS)*(1.0 - metal)

  diff_irradiance = SH9_irradiance(normal, sh9)
  diff_col = albedo / torch.pi * diff_irradiance.reshape(-1,3)

  if refl_probe is None:
    spec_col = SH_glossy_shade(normal, vdirs, rough, model_brdf, embed_fn, sh9, F0)
  elif only_spec:
    spec_col = spec_shade(normal, vdirs, rough, kS, refl_probe)
  else:
    rough_div = 0.2
    # choose render method accroding to the rough
    rough_sel = rough.flatten() < rough_div
    rough_usel = ~rough_sel

    normal_s = normal[rough_sel]
    vdirs_s = vdirs[rough_sel]
    rough_s = rough[rough_sel]
    normal_r = normal[rough_usel]
    vdirs_r = vdirs[rough_usel]
    rough_r = rough[rough_usel]
    #############################################
    spec_col = torch.ones_like(normal)
    if normal_s.shape[0] > 0:
      spec_col[rough_sel] = spec_shade(normal_s, vdirs_s, rough_s/rough_div, kS, refl_probe)
    if normal_r.shape[0] > 0:
      spec_col[rough_usel] = SH_glossy_shade(normal_r, vdirs_r, rough_r, model_brdf, embed_fn, sh9, F0)

  radiance = kD * diff_col + spec_col

  if clamp01:
    radiance = torch.clamp(radiance, 0., 1.)
  else:
    radiance = F.relu(radiance)
  return radiance


# sg: ..., 7
def SGProduct(sg1, sg2):
  lm = sg1[..., 3:4]+sg2[..., 3:4]
  um = (sg1[..., 3:4]*sg1[...,:3]+sg2[..., 3:4]*sg2[...,:3]) / lm
  
  umLength = torch.norm(um, dim = -1, keepdim=True)

  res = torch.ones_like(sg1)
  res[...,:3] = um * (1. / umLength)
  res[...,3:4] = lm * umLength
  res[...,-3:] = sg1[...,-3:]*sg2[...,-3:] * torch.exp(lm * (umLength - 1))
  return res

# sgs: ...,7
# normal: ...,3
def SGHemisphereIntegral(sgs, normal): 
  cos_beta = torch.sum(sgs[...,:3]*normal, -1, keepdim=True) #...,1
  lambda_val = torch.clip(sgs[...,3:4], min=EPS)

  inv_lambda_val = 1. / lambda_val
  t = torch.sqrt(lambda_val) * (1.6988 + 10.8438 * inv_lambda_val) / (
              1. + 6.2201 * inv_lambda_val + 10.2415 * inv_lambda_val * inv_lambda_val)

  inv_a = torch.exp(-t)
  mask = (cos_beta >= 0).float()
  inv_b = torch.exp(-t * torch.clamp(cos_beta, min=0.))
  s1 = (1. - inv_a * inv_b) / (1. - inv_a + inv_b - inv_a * inv_b)
  b = torch.exp(t * torch.clamp(cos_beta, max=0.))
  s2 = (b - inv_a) / ((1. - inv_a) * (b + 1.))
  s = mask * s1 + (1. - mask) * s2

  A_b = 2. * np.pi / lambda_val * (torch.exp(-lambda_val) - torch.exp(-2. * lambda_val))
  A_u = 2. * np.pi / lambda_val * (1. - torch.exp(-lambda_val))

  inte_res = A_b * (1. - s) + A_u * s # px,lx,1
  return inte_res*sgs[...,-3:]# px,lx,3

# sgs: px,lx,7
# normal: px,3
def SGIrradiance(sgs, normal, sum = True):
  cosSG = torch.ones(normal.shape[0], 7)
  cosSG[:,:3] = normal
  cosSG[:,3:4] *= 0.0315
  cosSG[:,-3:] *= 32.7080
  cosSG = cosSG.unsqueeze(1).expand_as(sgs)
  normal = normal.unsqueeze(1).expand(sgs.shape[0], sgs.shape[1], 3)

  lcosSG = SGProduct(sgs, cosSG) # px,lx,7
  irradiance = SGHemisphereIntegral(lcosSG, normal) - \
    31.7003 * SGHemisphereIntegral(sgs, normal) # px,lx,3
  if sum:
    irradiance = torch.sum(irradiance, 1) # px,3
  return F.relu(irradiance) # px,lx,3

# lSGs: px,lx,7, has decayed
@torch.no_grad()
def SG_render_core(albedo, metal, rough, normal, vdirs, lSGs, clamp01, self_shadow = True, refl_probe = None, only_spec = False):
  vdirs = -vdirs
  normal = normal/torch.norm(normal, dim=-1, keepdim=True)
  # DistributionTermSG and warp
  D_sg = torch.ones(normal.shape[0], 7)
  D_sg[:,:3] = reflect_dir(normal, vdirs)
  m2 = rough**2
  D_sg[:,3:4] = 2 / m2 / (4*pos_dot_eps(normal, vdirs))
  D_sg[:,-3:] *= 1 / (torch.pi * m2) # px,7
  
  if self_shadow:
    D_sg_ex = D_sg.unsqueeze(1).expand(lSGs.shape[0], lSGs.shape[1], 7) # px,lx,7
    ldSGs = SGProduct(D_sg_ex, lSGs) # px,lx,7

    specIrr = SGIrradiance(ldSGs, normal) # px,3
    diffIrr = SGIrradiance(lSGs, normal)
  
  else:
    D_sg_ex = D_sg.unsqueeze(1).expand(normal.shape[0], lSGs.shape[0], 7) # px,lx,7
    lSGs_ex = lSGs.unsqueeze(0).expand_as(D_sg_ex)
    ldSGs = SGProduct(D_sg_ex, lSGs_ex) # px,lx,7

    specIrr = SGIrradiance(ldSGs, normal) # px,3
    diffIrr = SGIrradiance(lSGs_ex, normal)
  

  wo = vdirs
  wi = D_sg[:,:3]
  wh = normal
  NdotV = pos_dot(normal, wo)
  NdotL = NdotV # pos_dot_eps(normal, wi)

  F0 = get_F0(metal, albedo)
  _F = fresnelSchlick(F0, NdotV)
  #G = GeometrySchlickGGX(NdotV, rough)**2
  G = 1.0/(GeometryBlender(NdotV, rough)*2+1)

  Moi = _F * G / (4 * NdotL * NdotV + EPS)

  diff_a = 1.0
  spec_a = 1.0 ############################## WARNING HACK 

  spec_col = Moi*specIrr *spec_a
  diff_col = albedo / torch.pi * diffIrr *diff_a

  kS = fresnelSchlickRoughness(F0, NdotV, rough); 
  kD = (1. - kS)*(1. - metal)

  radiance = kD*diff_col + spec_col

  if clamp01:
    radiance = torch.clamp(radiance, 0., 1.)
  else:
    radiance = F.relu(radiance)
  return radiance