import time, sys, os
import torch
from torchvision.transforms.functional import gaussian_blur
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from opt import get_opts
import numpy as np
from einops import rearrange

import open3d as o3d
from tqdm import tqdm

from datasets import dataset_dict
from datasets.ray_utils import get_ray_directions, get_rays
from models.networks import NGP
from models.rendering import render, render_surface_normal, render_surface_rgb
from train import depth2img
from utils import load_ckpt

from insert_utils import *
from insert_models import *
from global_light import *
from server import *
from shadow_fields import *
from sg_shadow import *
from envfit import *
import struct

import warnings; warnings.filterwarnings("ignore")
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'


SH_order = 3
use_std_sf = True
use_sg_base = True
sg_use_self_shadow = True
Viewer_bin_path = '/home/lofr/Projects/Render/objViewer/bin/'
Viewer_sf_path = '/home/lofr/Projects/Render/objViewer/bin/sf/results/'
Viewer_sg_path = '/home/lofr/Projects/Render/objViewer/bin/sg/results/'
IRAdobe_path = '/home/lofr/Projects/NeRF/InverseRenderingOfIndoorScene-master/results/'
EMLight_path = '/home/lofr/Projects/Light/EMLight-master/RegressionNetwork/'
global_counter = 0
dep_adj_final = 0

class NGP_Insertor():
  def __init__(self, hparams):
    self.hparams = hparams
    rgb_act = 'None' if (self.hparams.use_exposure  or self.hparams.use_EXR) else 'Sigmoid'
    self.model = NGP(scale=hparams.scale, rgb_act=rgb_act, use_raw_HDR=self.hparams.use_EXR).cuda()
    load_ckpt(self.model, hparams.ckpt_path)

    self.gen_path = os.path.join('./insert/generate/', hparams.exp_name)
    self.has_pc = os.path.exists(os.path.join(self.gen_path, 'pc.ply'))
    self.has_sur = os.path.exists(os.path.join(self.gen_path, 'surface.npy'))
    self.has_sh = os.path.exists(os.path.join(self.gen_path, 'mat_sh_000049.tar'))
    if self.has_sh or self.has_sur:
      read_meta = False
    else:
      read_meta = True

    kwargs = {'root_dir': hparams.root_dir,
            'downsample': hparams.downsample,
            'read_meta': read_meta}
    if hparams.use_EXR and \
      (hparams.dataset_name == 'colmap_exr' or hparams.dataset_name == 'myblender'):
      kwargs.update({'use_EXR': True})
    dataset = dataset_dict[hparams.dataset_name](**kwargs)

    l_resol = hparams.low_resolution
    self.K = dataset.K
    self.K[:2] = self.K[:2] / l_resol
    self.W = int(dataset.img_wh[0] / l_resol)
    self.H = int(dataset.img_wh[1] / l_resol)
    self.directions = get_ray_directions(self.H, self.W, self.K, device='cuda').reshape(self.H,self.W,3)
    self.screen_bound = [[0,0], [self.H,self.W]]
    self.dataset = dataset
    self.SH_ray_dirs = None
    self.SH_ray_dirs_vis = None
    self.global_SH = torch.zeros(1,SH_order**2,3).cuda() ##

    self.last_depth = None
    self.last_rgb = None

    embed_fn_v, input_ch_v = get_embedder(3)
    model_brdf = MLP(input_ch=input_ch_v*2+1, output_ch=2*SH_order**2, D = 2, W = 128, skips=[])
    model_brdf.load_state_dict(torch.load('./insert/data/model_brdf{}.tar'.format(SH_order))['model'])
    model_brdf.eval()
    self.model_brdf = model_brdf
    self.embed_fn_v = embed_fn_v

    self.sf = None#SimplifySF()
    self.sg_shadow = None
    self.env_opt = EnvOptim()

    os.makedirs(self.gen_path, exist_ok = True)

  def set_sf(self, sf_path):
    self.sf = ComplexSF(sf_path, SH_order**2)
  def set_sg_shadow(self, pca_path):
    self.sg_shadow = SGShadow(pca_path, 20, 128, 2, envH = 74, envW = 148)  ####
    #self.sg_shadow = SGShadow(pca_path, 40, 128, 4, envH = 128, envW = 128)  ####
  
  def render(self, rays_o, rays_d, 
    toCPU = True, toNumpy = True, **kwargs):
    if self.hparams.dataset_name in ['colmap', 'nerfpp']:
        exp_step_factor = 1/256
    else: exp_step_factor = 0
    t = time.time()

    kwargs.update({
      'test_time': True,
      'to_cpu': toCPU, 'to_numpy': toNumpy,
      'T_threshold': 1e-2,
      'exposure': torch.cuda.FloatTensor([1.0]), #
      'max_samples': 100,
      'exp_step_factor': exp_step_factor,
      "val_batch_size": self.hparams.val_batch_size}
    )
    results = render(self.model, rays_o, rays_d, **kwargs)
    self.dt = time.time()-t
    if kwargs.get('return_full_res', False):
      return results
    #self.mean_samples = results['total_samples']/len(rays_o)
    return results["rgb"], results["depth"]

  def render_pose(self, pose, toCPU = True, toNumpy = True, **kwargs):
    directions = get_ray_directions(self.H, self.W, self.K, device='cuda')
    rays_o, rays_d = get_rays(directions, pose.cuda())
    # tts = torch.concat([rays_o[:1], rays_d], 0)
    # rrg = torch.ones_like(tts)
    # write2ply(rrg.cpu().numpy(), tts.cpu().numpy(), './insert/tt.ply')

    rgb, depth = self.render(rays_o, rays_d, toCPU, toNumpy, **kwargs)
    rgb = rearrange(rgb, "(h w) c -> h w c", h=self.H)
    depth = rearrange(depth, "(h w) -> h w", h=self.H)
    torch.cuda.synchronize()
    if toCPU:
      rays_o = rays_o.cpu(); rays_d = rays_d.cpu()
    if toNumpy:
      rays_o = rays_o.numpy(); rays_d = rays_d.numpy()

    return rgb, depth, rays_o, rays_d

  def generate_surface(self, visualize = False, save = False):
    save_path = os.path.join(self.gen_path, "surface.npy")
    if self.has_sur:
      surface_info = np.load(save_path, allow_pickle=True).item()
      self.rgbs = surface_info['rgbs']
      self.spts = surface_info['spts']
      self.normals = surface_info['normals']
      return
    rgbs = []; pts = []; normals = []
    kwargs = {}
    if self.hparams.use_EXR:
      kwargs = {'output_radiance': True}
    for pose in tqdm(self.dataset.poses, 'generate surface points'):
      rgb, depth, rays_o, rays_d = self.render_pose(pose, False, False, **kwargs)
      rays_o = rays_o.reshape(self.H, self.W,3)
      rays_d = rays_d.reshape(self.H, self.W,3)
      depth = depth.reshape(self.H, self.W,1)
      surface_pts = rays_o + depth*rays_d
      #pts_normal = pts2normal(surface_pts.unsqueeze(0))[0]

      #tps = torch.rand(800,800,3)-0.5
      #tttn = render_surface_normal(self.model, tps)
      pts_normal = render_surface_normal(self.model, surface_pts)#

      rgbs.append(rgb.cpu().numpy())
      pts.append(surface_pts.cpu().numpy())
      normals.append(pts_normal.cpu().numpy())
      if visualize:
        rrr = rgbs[-1]
        if self.hparams.use_EXR:
          rrr = rrr/(1+rrr)
          rrr = torch.pow(rrr, 1.0/2.2)
        show_im(rrr, False)
        show_im(normals[-1]*0.5+0.5, False)
    self.rgbs = np.stack(rgbs, 0)
    self.spts = np.stack(pts, 0)
    self.normals = np.stack(normals, 0)
    self.has_sur = True
    if save:
      np.save(save_path, {
        "rgbs": self.rgbs,
        "spts": self.spts,
        "normals": self.normals
      })

  def generate_envmaps(self, env_num = 512):
    res_save_path = os.path.join(self.gen_path, 'envmaps.npy')
    if os.path.exists(res_save_path):
      return
    save_path = os.path.join(self.gen_path, "surface.npy")
    if self.has_sur:
      surface_info = np.load(save_path, allow_pickle=True).item()
      spts = surface_info['spts'].reshape(-1, 3)
    else:
      self.generate_point_cloud(save = True)
      spts = self.spts.reshape(-1, 3)
    shuffle_idx = np.random.permutation(spts.shape[0])
    spts = spts[shuffle_idx]
    spts = spts[:env_num]
    print('Generate env maps ...')
    envmaps = [self.generate_probe(torch.Tensor(pt), return_envmap=True) for pt in tqdm(spts)]
    envmaps = np.stack(envmaps, 0) # num, 128, 128, 3
    np.save(res_save_path, envmaps)


  def load_or_train_envmaps(self, epoch = 1001):
    self.env_model = EnvTrainer(self.hparams.exp_name, epoch=epoch)
    self.env_model.train(epoch, True)
    self.env_model.eval_mode()


  def generate_point_cloud(self):
    if self.has_pc:
      binfo = np.load(os.path.join(self.gen_path, "btrans.npy"), allow_pickle=True).item()
      self.blender_trans = binfo['trans']
      self.blender_scale = binfo['scale']
      return
    self.generate_surface(save=True)
    rgbs = self.rgbs.reshape(-1,3)
    pts = self.spts.reshape(-1,3)
    shuffle_idx = np.random.permutation(pts.shape[0])
    rgbs = rgbs[shuffle_idx]
    pts = pts[shuffle_idx]
    max_pts_num = self.hparams.max_pc_pts_num
    rgbs = rgbs[:max_pts_num]
    pts = pts[:max_pts_num]

    if self.hparams.use_EXR:
      rgbs = rgbs / (1+rgbs)
      rgbs = np.power(rgbs, 1.0/2.2)

    save_path = os.path.join(self.gen_path, "pc.ply")
    write2ply(rgbs, pts, save_path)
    binfo = {
      'trans': self.dataset.blender_trans.astype(np.float32),
      'scale': self.dataset.blender_scale
    }
    self.blender_trans = binfo['trans']
    self.blender_scale = binfo['scale']
    np.save(os.path.join(self.gen_path, "btrans.npy"), binfo, allow_pickle=True)
    self.has_pc = True

  def train_global_SH_light(self, visualize_global_SH = False):
    self.generate_surface(save=True)
    gle = GlobalLightEstimator(self.gen_path)
    if not gle.calc_complete:
      gle.detect_planar_patch()
      gle.save_results(self)
    '''
    kwargs = {
      'iters': 200,
      'ckpt_save': 199,
      'batch': 20480*16,
      'mat_smooth_range': 1e-2,
      'mat_smooth_weight': 0.2,
      'lrate': 1e-4,
      'lrate_decay': 300,
      'nerf_model': None,
      'hdr_mapping': self.hparams.train_SH_HDR_mapping
    }
    self.global_SH = \
      train_global_env_prec(gle.t_pts, gle.t_normal, gle.t_rgbs, 
        None, None,
        self.gen_path, SH_order**2, True, **kwargs
      )
    '''
    kwargs = {
      'iters': 200,
      'ckpt_save': 199,
      'batch': 20480*16,
      'mat_smooth_range': 1e-2,
      'mat_smooth_weight': 0.2,
      'lrate': 1e-4,
      'lrate_decay': 2000,#300,
      'nerf_model': self,#None,
      'hdr_mapping': self.hparams.train_SH_HDR_mapping,
      #'downsample_pts_num': 4096*5
    }
    self.global_SH = \
      train_global_env_prec(gle.t_pts, gle.t_normal, gle.t_rgbs, 
        gle.t_rgb_shs, gle.t_opc_shs,
        self.gen_path, SH_order**2, True, **kwargs
      )
    
    
    '''
    self.global_SH = \
      train_global_env(gle.t_pts, gle.t_normal, gle.t_rgbs, 
        self.gen_path, SH_order**2, True, **kwargs
      )
    '''
    
    if visualize_global_SH:
      visualize_SH(self.global_SH, 48)
  
  # pt: 3
  # return_envmap = True, will only return CPU envmap
  def generate_probe(self, pt, SH_probe = True, 
    visualize_probe = False, return_envmap = False, 
    use_sphere_rays_sample = False):

    if self.SH_ray_dirs == None:
      if use_sphere_rays_sample:
        self.SH_ray_dirs = get_sphere_rays(1, 2048)
      else:
        self.SH_ray_dirs = get_cubemap_rays(1, 32) 
    #self.SH_ray_dirs = get_sphere_rays(1, 2048)
    ray_dirs = self.SH_ray_dirs
    
    ray_dirs = ray_dirs.reshape(-1, 3)
    rays_o = pt.unsqueeze(0).expand_as(ray_dirs)
    kwargs = {'SH_bkg': self.global_SH}
    if self.hparams.use_EXR:
      kwargs.update({'output_radiance': True})

    rgb, _ = self.render(rays_o, ray_dirs, False, False, **kwargs)

    if self.hparams.gen_probe_HDR_mapping:
      rgb = rgb/(1+rgb)
      rgb = torch.exp(rgb, 1.0/2.2)

    self.cubemap_rgb = rgb

    if return_envmap:
      return cubemap2env_map(rgb, 32, 128, 128).cpu().numpy()

    if SH_probe:
      sh_coeff = get_SH_coeff(ray_dirs[None,...], rgb[None,...])
    
      if visualize_probe:
        visualize_SH(sh_coeff, 48, use_cv=True)
        #visualize_env(torch.stack([ray_dirs, rgb], 0), 256, use_cv=True) 
        visualize_env_rad(torch.stack([ray_dirs, rgb], 0), 32) 
        #show_im_cv(cubemap2env_map(rgb, 32, 128, 128).cpu().numpy())
      return sh_coeff
    # SG probe
    else:
      envmap = cubemap2env_map(rgb, 32, 128, 128)
      #sgs = self.env_model.eval(envmap)
      sgs = self.env_opt.eval(envmap)
      if visualize_probe:
        visualize_env(torch.stack([ray_dirs, rgb], 0), 32, use_cv=True) 
        show_im_cv(SG2Envmap_forDraw(sgs, 128, 128).cpu().numpy())
      return sgs 

  # pt: x,3
  def generate_SH_probes(self, pt, return_raw_rgb = False):
    pt_num = pt.shape[0]
    ray_dirs = get_sphere_rays(pt_num, 2048) # x,2048,3
    
    rays_o = pt.unsqueeze(1).expand_as(ray_dirs)

    kwargs = {'SH_bkg': self.global_SH}
    if self.hparams.use_EXR:
      kwargs.update({'output_radiance': True})

    rgb, _ = self.render(
      rays_o.reshape(-1,3), 
      ray_dirs.reshape(-1,3), 
      False, False, **kwargs) # x*2048,3

    if self.hparams.gen_probe_HDR_mapping:
      rgb = rgb/(1+rgb)
      rgb = torch.exp(rgb, 1.0/2.2)

    rgb = rgb.reshape(pt_num, -1, 3)
    if return_raw_rgb:
      return rgb, ray_dirs

    sh_coeff = get_SH_coeff(ray_dirs, rgb) #x,9,3
    return sh_coeff    

  # only used for optimize
  def generate_SH_probes_for_precompute(self, pt):
    pt_num = pt.shape[0]
    ray_dirs = get_sphere_rays(pt_num, 2048) # x,2048,3
    
    rays_o = pt.unsqueeze(1).expand_as(ray_dirs)

    kwargs = {
      'return_full_res': True,
      'blend_bkg': False
    } # no global light, keep bkg black
    if self.hparams.use_EXR:
      kwargs.update({'output_radiance': True})

    results = self.render(
      rays_o.reshape(-1,3), 
      ray_dirs.reshape(-1,3), 
      False, False, **kwargs) # x*2048,3
    rgb = results['rgb']
    opacity = results['opacity']

    rgb = rgb.reshape(pt_num, -1, 3)
    opacity = 1.0 - opacity.reshape(pt_num, -1, 1)

    rgb_sh_coeff = get_SH_coeff(ray_dirs, rgb) #x,9,3
    opacity_sh_coeff = get_SH_coeff(ray_dirs, opacity) #x,9,1
    return rgb_sh_coeff, opacity_sh_coeff


  def enlarge_range(self, bbox, scale): 
    dH = bbox[1][0] - bbox[0][0]
    dW = bbox[1][1] - bbox[0][1]
    nbbox = [
      [int(max(0,bbox[0][0]-scale*dH)), int(max(0,bbox[0][1]-scale*dW))],
      [int(min(self.H, bbox[1][0]+scale*dH)), int(min(self.W, bbox[1][1]+scale*dW))]
    ]
    return nbbox

  def shadow_field(self, rays_o, rays_d, rgb, depth_sur, model_sh9, **kwargs):
    model_r = kwargs.get('model_radius', None)
    model_pos = kwargs.get('model_pos', None)
    if model_r == None or model_pos == None:
      print('Use shadow field, but infos not complete!')
      return

    rays_o = rays_o.reshape_as(rgb)
    rays_d = rays_d.reshape_as(rgb) # H,W,3
    
    pts = (rays_o + rays_d * depth_sur).reshape(-1, 3)
    #sel_ptsn = pts2normal(sel_pts.unsqueeze(0))[0]
    #sel_pts = sel_pts.reshape(-1, 3); sel_ptsn = sel_ptsn.reshape(-1, 3)
    model_rot_inv = kwargs.get("model_rot_inv")
    if model_rot_inv is not None:
      smap = soft_shadow_map(self.sf, model_pos, model_r, 
        rotate_SH_by_recalc(self.SH_ray_dirs[0], self.cubemap_rgb, model_rot_inv), 
        pts, model_rot_inv)
    else:
      smap = soft_shadow_map(self.sf, model_pos, model_r, model_sh9, pts)
    smap = smap.reshape(rgb.shape[0], rgb.shape[1], 1)
    ######
    
    # cons = depth_sur.flatten() == 0
    # cons_smap = smap.flatten()[cons]
    # smap_blur = gaussian_blur(smap.permute(2,0,1), (9,9))
    # smap_blur.flatten()[cons] = cons_smap
    # smap = smap_blur.permute(1,2,0)
    ######
    return rgb*smap

  def shadow_cast(self, rays_o, rays_d, rgb, depth_sur, VP, texSize, s_map, model_r):
    #show_im(s_map)
    rays_o = rays_o.reshape_as(rgb)
    rays_d = rays_d.reshape_as(rgb) # H,W,3
    
    pts = (rays_o + rays_d * depth_sur).reshape(-1, 3)
    pts_n = torch.concat([pts, torch.ones(pts.shape[0], 1)], -1)
    pts_ras = (VP @ pts_n.T).T
    pts_ras[:,:3] /= pts_ras[:, -1:]
    ras_x = torch.clamp(((pts_ras[:,0]+1.0)/2.0 * texSize).long(), 0, texSize-1)
    ras_y = torch.clamp(((-pts_ras[:,1]+1.0)/2.0 * texSize).long(), 0, texSize-1)
    ras_z = 0.5*(pts_ras[:,2] + 1)
    #show_im(ras_z.reshape(rays_o.shape[0], rays_o.shape[1], 1))
    shadow_dis = ras_z - s_map[ras_y, ras_x, 0]
    out_shadow = shadow_dis < 0
    shadow_d = ((shadow_dis / (model_r*50))**2).clip(min = 0.2, max = 1.0)

    smap = torch.ones(rays_o.shape[0], rays_o.shape[1], 1)
    smap.flatten()[:] = shadow_d
    smap.flatten()[out_shadow] = 1.0
    ##########
    smap_blur = gaussian_blur(smap.permute(2,0,1), (9,9))
    smap = smap_blur.permute(1,2,0)

    return rgb * smap

  def ssdf_shadow(self, rays_o, rays_d, rgb, depth_sur, lSGs, **kwargs):
    model_r = kwargs.get('model_radius', None)
    model_pos = kwargs.get('model_pos', None)
    if model_r == None or model_pos == None:
      print('Use ssdf shadow, but infos not complete!')
      return

    rays_o = rays_o.reshape_as(rgb)
    rays_d = rays_d.reshape_as(rgb) # H,W,3
    
    pts = (rays_o + rays_d * depth_sur).reshape(-1, 3)
    model_rot_inv = kwargs.get("model_rot_inv", None)
    #pts = torch.Tensor([100,0,0]).reshape(-1,3)
    if model_rot_inv is not None:
      lSGs_rot = lSGs.clone()
      lSGs_rot[:, :3] = (model_rot_inv @ lSGs_rot[:, :3].T).T
      smap = self.sg_shadow.calc_shadow_factor(model_r, pts, model_pos, lSGs_rot, model_rot_inv)
    else:
      smap = self.sg_shadow.calc_shadow_factor(model_r, pts, model_pos, lSGs)

    smap = smap.reshape(rgb.shape[0], rgb.shape[1], 1)
    #show_im(smap*rgb)
    ######
    
    # cons = depth_sur.flatten() == 0
    # cons_smap = smap.flatten()[cons]
    # smap_blur = gaussian_blur(smap.permute(2,0,1), (9,9))
    # smap_blur.flatten()[cons] = cons_smap
    # smap = smap_blur.permute(1,2,0)
    ######
    return rgb*smap

  def render_object(
    self, model_bbox_cur, normals, depths, 
    sh_or_sg, pose, metal = 0.9, 
    rough = 0.2, albedo = None, **kwargs):

    mask = (depths > 1e-6).flatten()

    normal_msk = normals.reshape(-1,3)[mask]
    if albedo is None:
      albedo = torch.ones_like(normal_msk)
    elif albedo.shape[0] == 1:
      albedo = albedo.expand_as(normal_msk)
    else:
      albedo = albedo.reshape(-1,3)[mask]
    if type(metal) == float:
      metal = torch.ones(normal_msk.shape[0],1)*metal
    else:
      metal = metal.reshape(-1,1)[mask]
    if type(rough) == float:
      rough = torch.ones(normal_msk.shape[0],1)*rough
    else:
      rough = rough.reshape(-1,1)[mask].clip(0.2,1.0)

    hs = model_bbox_cur[0][0]
    ws = model_bbox_cur[0][1]
    hl = model_bbox_cur[1][0]
    wl = model_bbox_cur[1][1]
    height = hl - hs; width = wl - ws

    #directions = get_ray_directions(self.H, self.W, self.K, device='cuda')
    rays_o, rays_d = get_rays(self.directions[hs:hl, ws:wl,:].reshape(-1,3), pose.cuda())
    vdirs = normalize(rays_d.reshape(-1,3)[mask])

    global use_sg_base
    global sg_use_self_shadow
    if use_sg_base:
      lSGs = sh_or_sg
      if sg_use_self_shadow:
        rays_o = rays_o.reshape(-1,3)[mask]
        _dep = depths.reshape(-1,1)[mask]
        pts = rays_o + _dep * vdirs
        
        #pts += normal_msk * 0.004
        lSGs = self.sg_shadow.calc_self_shadow_light_dacay(
          kwargs.get('model_radius'), pts, kwargs.get('model_pos'), 
          sh_or_sg, kwargs.get("model_rot_inv")
        )
        #pt = torch.Tensor([0.2223, -0.0500, -0.3564])
        #self.sg_shadow._deubug(pt, sh_or_sg[0, :3], 
        #kwargs.get('model_radius'), kwargs.get('model_pos'), kwargs.get("model_rot_inv"))
      
      cols = SG_render_core(
        albedo, metal, rough, normal_msk, vdirs, lSGs, 
        (not self.hparams.render_HDR_mapping),
        sg_use_self_shadow, self.cubemap_rgb
      )
      #cols = ssdf[:,:1].expand_as(cols)*0.5+0.5
    else:
      cols = SH_render_core(
        albedo, metal, rough, normal_msk, vdirs, sh_or_sg, 
        self.embed_fn_v, self.model_brdf, (not self.hparams.render_HDR_mapping),
        self.cubemap_rgb
      )
    
    render_res_t = torch.zeros(height*width, 3)
    render_res_t[mask] = cols
    render_res_t = render_res_t.reshape(height, width, 3)
    render_res = torch.zeros(self.H, self.W, 3)
    render_res[hs:hl, ws:wl,:] = render_res_t
    #show_im(render_res[...,0])

    depth_t = torch.zeros(self.H, self.W)
    depth_t[hs:hl, ws:wl] = depths
    return render_res, depth_t # H,W,3,   H,W

  def sg_render_res_handle(self, model_bbox_cur, sg_res, depths):
    hs = model_bbox_cur[0][0]
    ws = model_bbox_cur[0][1]
    hl = model_bbox_cur[1][0]
    wl = model_bbox_cur[1][1]
    render_res = torch.zeros(self.H, self.W, 3)
    render_res[hs:hl, ws:wl,:] = sg_res
    depth_t = torch.zeros(self.H, self.W)
    depth_t[hs:hl, ws:wl] = depths
    return render_res, depth_t # H,W,3,   H,W


  def getUpdateRange(self, bbox_cur, bbox_last):
    if bbox_last is None or bbox_cur is None:
      return self.screen_bound
    else:
      return [
        [min(bbox_cur[0][0], bbox_last[0][0]),
        min(bbox_cur[0][1], bbox_last[0][1])],
        [max(bbox_cur[1][0], bbox_last[1][0]),
        max(bbox_cur[1][1], bbox_last[1][1])]
      ]

  # if use SG base, the normal param is the final object render result
  def render_insert_object(self, normals, depths, pose, SHorSG, metal = 0.9, rough = 0.2, albedo = None, full_return = False, **kwargs):
    
    model_bbox = kwargs.get('model_bbox', None)
    model_bbox_last = kwargs.get('model_bbox_last', None)
    '''
    if use_sg_base:
      render_res, depth_t = self.sg_render_res_handle(model_bbox, normals, depths)
    else:
      render_res, depth_t = self.render_object(model_bbox, normals, depths, SHorSG, pose, metal, rough, albedo, **kwargs)
    '''
    render_res, depth_t = self.render_object(model_bbox, normals, depths, SHorSG, pose, metal, rough, albedo, **kwargs)

    updata_range = self.getUpdateRange(model_bbox, model_bbox_last)
    hs = updata_range[0][0]
    ws = updata_range[0][1]
    hl = updata_range[1][0]
    wl = updata_range[1][1]
    height = hl - hs; width = wl - ws

    rays_o, rays_d = get_rays(self.directions[hs:hl, ws:wl,:].reshape(-1,3), pose.cuda())
    render_res_clip = render_res[hs:hl, ws:wl,:].reshape(-1,3)
    depth_t_clip = depth_t[hs:hl, ws:wl].flatten()

    #show_im(render_res.reshape(self.H,self.W,3))
    gen_shadow = kwargs.get('gen_shadow')
    kwargs_r = {'IM_bkg': render_res_clip, 'mesh_depth_map': depth_t_clip.flatten()}
    if self.hparams.use_EXR:
      kwargs_r.update({'output_radiance': True})
    
    rgb, depth_sur = self.render(rays_o, rays_d, False, False, **kwargs_r)
    rgb = rgb.reshape(height, width, 3)
    depth_sur = depth_sur.reshape(height, width, 1)
    if self.last_rgb is not None:
      self.last_rgb[hs:hl, ws:wl,:] = rgb
      self.last_depth[hs:hl, ws:wl, :] = depth_sur
    else:
      self.last_rgb = rgb
      self.last_depth = depth_sur
    
    rgb = self.last_rgb
    depth_sur = self.last_depth

    if gen_shadow != 0:
      rays_o, rays_d = get_rays(self.directions.reshape(-1,3), pose.cuda())
      # for shadow cast, SG and SH is the same
      if gen_shadow == 2:
        s_texSize = kwargs.get('s_texSize', None)
        rgb = self.shadow_cast(
          rays_o, rays_d, rgb, depth_sur, 
          kwargs.get('s_VP'), s_texSize, kwargs.get('s_im'), kwargs.get('model_radius'))
      else:
        if use_sg_base:
          rgb = self.ssdf_shadow(rays_o, rays_d, rgb, depth_sur, SHorSG, **kwargs)
        else:
          rgb = self.shadow_field(rays_o, rays_d, rgb, depth_sur, SHorSG, **kwargs)

    rgb_final = rgb
    if self.hparams.render_HDR_mapping:
      rgb_final = rgb_final / (1+rgb_final)
      rgb_final = torch.pow(rgb_final, 1.0/2.2)
    rgb_final = rgb_final.cpu().numpy()

    if full_return:
      return rgb_final, rgb, depth_t, render_res

    return rgb_final


class NGP_Server:
  def __init__(self, insertor, record = False):
    self.insertor = insertor
    self.server = Server('127.0.0.1', 5001)
    HWF = [insertor.H, insertor.W, insertor.K[0,0].item()]
    self.server.send(struct.pack('iif', *HWF))
    self.server.send(self.insertor.blender_trans.tobytes())
    self.server.send(struct.pack('f', self.insertor.blender_scale))
    print('***********************')
    print('H,W,F for current scene is: ', HWF)
    print('***********************')
    self.act_dict = {
      1: self.ProbePosDecoder,
      2: self.CamPoseDecoder,
      3: self.MapDecoder,
      4: self.MaterialDecoder,
      5: self.ShadowFieldDecoder,
      6: self.Render,
      7: self.ShadowMapDecoder,
      8: self.ShadowPathDecoder,
      9: self.SSDFPathDecoder,
      10: self.SG_use_sshadow,
      11: self.CmpMethodsDecoder,
      12: self.RunDecompositionCmpDecoder,
      13: self.UpdateSaveIndexDecoder
    }
    self.cam_pose = None
    self.normal = None
    self.depth = None
    self.sh = None
    '''
    torch.Tensor([[ 3.5835,  4.7817,  5.6890],
        [ 0.0600,  0.0800,  0.0952],
        [ 2.6326,  3.5128,  4.1794],
        [ 2.2019,  2.9381,  3.4956],
        [ 0.1036,  0.1382,  0.1644],
        [-0.6205, -0.8279, -0.9850],
        [ 0.9091,  1.2131,  1.4432],
        [ 2.5738,  3.4345,  4.0861],
        [ 2.1180,  2.8263,  3.3625]]).reshape(1,9,3)
    '''
    self.sg = None
    self.fixed_lighting = False # if sh is fixed, than sh should be assigned first

    # for shadow
    self.model_pos = None
    self.model_radius = None
    self.model_rot_inv = None
    self.model_bbox = None
    self.model_bbox_last = None
    self.pose_last = None

    self.s_texSize = None
    self.s_VP = None
    self.s_im = None

    self.render_num = 0
    self.last_render_num = -1
    self.save_idx = 0

    self.metal = 0.9
    self.rough = 0.2
    self.albedo = None
    self.dt = 0
    self.vw = None
    if record:
      video_path = os.path.join(insertor.gen_path, 'video.avi')
      fourcc = cv2.VideoWriter_fourcc(*'XVID')
      self.vw = cv2.VideoWriter(video_path, fourcc, 10.0, (HWF[1], HWF[0]),True)

  def MainDirectionLightSender(self):
    #self.main_light = get_SH_main_direction(self.sh)#
    #self.main_light = get_cubemap_main_direction(
    #  self.insertor.cubemap_rgb, self.insertor.SH_ray_dirs)
    ################################################
    t = torch.Tensor([0.194, -0.165, -0.270]) - self.model_pos
    self.main_light = normalize(t.reshape(1,3))
    ################################################
    main_light_np =  self.main_light.flatten().cpu().numpy().astype(np.float32) 
    self.server.send(main_light_np.tobytes())
    pass

  def SGLightSender(self):
    sgs = self.sg.cpu().numpy().astype(np.float32)
    self.server.send(sgs.tobytes())

  def ProbePosDecoder(self, buf):
    visualize = False
    if self.last_render_num < self.render_num:
      self.last_render_num = self.render_num
    else:
      visualize = True
      self.model_bbox_last = None
    self.shadow_mode, px,py,pz = struct.unpack('ifff', buf[:16])
    self.model_rot_inv = torch.Tensor(np.frombuffer(buf[16:], np.float32)).reshape(3,3).T
    self.model_pos = torch.Tensor([px,py,pz]).cuda()
    if use_sg_base:
      if not self.fixed_lighting:
        self.sg = self.insertor.generate_probe(self.model_pos, False, visualize)
        self.sg = trans_raw_sg(self.sg)
        #self.SGLightSender()
    else:
      if not self.fixed_lighting:
        self.sh = self.insertor.generate_probe(self.model_pos, True, visualize)
        # if visualize:
        #   try:
        #     self.server.send(struct.pack('i', 0)) # send 0 to client, render complete
        #   except:
        #     pass
    ##########################
    if self.shadow_mode == 2:
      self.MainDirectionLightSender()
    #ray_rgb = torch.concat([self.main_light, torch.Tensor([1,0,0]).reshape(1,3)], 0)
    #visualize_env(ray_rgb, 48, use_cv=True, cv_name='123')

  def CamPoseDecoder(self, buf):
    pose = struct.unpack('f'*16, buf)
    pose = torch.Tensor(pose).reshape(4,4)[:3].cuda()
    pose = torch.stack([pose[:,0],-pose[:,1],-pose[:,2], pose[:,3]], -1)
    self.cam_pose = pose
  '''
  def MapDecoder(self, buf):
    H,W = struct.unpack('ii', buf[:8])
    im = np.frombuffer(buf[8:], np.float32).reshape(H,W,4)
    normal = im[..., :3]
    depth = im[..., 3]
    self.normal = torch.Tensor(normal).cuda().flip(0)
    self.depth = torch.Tensor(depth).cuda().flip(0)
  '''
  def MapDecoder(self, buf):
    self.model_radius,hs,ws,hl,wl = struct.unpack('fiiii', buf[:20])
    self.model_bbox_last = self.model_bbox
    self.model_bbox = [[hs, ws], [hl, wl]]
    H = hl - hs; W = wl - ws
    im = np.frombuffer(buf[20:], np.float32)
    if im.shape[0] > H*W*4: # SV BRDF
      pixelNum = H*W*3
      normal = im[:pixelNum].reshape(H,W,3)
      albedo = im[pixelNum:pixelNum*2].reshape(H,W,3)
      depthMetalRough = im[pixelNum*2:].reshape(H,W,3)

      depth = depthMetalRough[..., 0]
      metal = depthMetalRough[..., 1]
      rough = depthMetalRough[..., 2]
      self.normal = torch.Tensor(normal).cuda().flip(0)
      self.depth = torch.Tensor(depth).cuda().flip(0)
      self.albedo = torch.Tensor(albedo).cuda().flip(0)
      self.metal = torch.Tensor(metal).cuda().flip(0)
      self.rough = torch.Tensor(rough).cuda().flip(0)
    else:
      im = im.reshape(H,W,4)
      normal = im[..., :3] # if use SG base, normal is the final result
      depth = im[..., 3]
      self.normal = torch.Tensor(normal).cuda().flip(0)
      self.depth = torch.Tensor(depth).cuda().flip(0)
    #show_im(self.normal*0.5+0.5)
    # show_im(self.albedo)
    # show_im(self.metal)
    # show_im(self.rough)

  def MaterialDecoder(self, buf):
    self.rough, self.metal, r, g, b = struct.unpack('fffff', buf)
    self.albedo = torch.Tensor([r,g,b]).reshape(1,3)

  def ShadowFieldDecoder(self, buf):
    r,hmin,wmin,hmax,wmax = struct.unpack('fiiii', buf)
    self.model_radius = r
    self.model_bbox = [[hmin, wmin], [hmax, wmax]]
 
  def ShadowMapDecoder(self, buf):
    texSize = struct.unpack('i', buf[:4])[0]
    s_vp = struct.unpack('f'*16, buf[4:68])
    s_vp = torch.Tensor(s_vp).reshape(4,4).cuda()
    #s_vp = torch.stack([s_vp[:,0],-s_vp[:,1],-s_vp[:,2], s_vp[:,3]], -1)
    s_im = np.frombuffer(buf[68:], np.float32).reshape(texSize, texSize, 1)
    s_im = torch.Tensor(s_im).cuda().flip(0)
    self.s_texSize = texSize
    self.s_VP = s_vp
    self.s_im = s_im
    #show_im(s_im)

  def ShadowPathDecoder(self, buf):
    model_name = buf.decode()
    sf_path = './insert/model_data/'+model_name+'.tar'
    raw_sf_path = os.path.join(Viewer_sf_path, model_name+'.txt')
    if not os.path.exists(sf_path):
      transform_sf_txt_to_torch(raw_sf_path, sf_path)
    self.insertor.set_sf(sf_path)

    global use_sg_base
    if use_sg_base:
      use_sg_base = False

  def SSDFPathDecoder(self, buf):
    model_name = buf.decode()
    sg_path = os.path.join(Viewer_sg_path, model_name+'.tar')
    self.insertor.set_sg_shadow(sg_path)

    global use_sg_base
    if not use_sg_base:
      use_sg_base = True

  def CompositeCmpResults(self, sg_res, env_im, subfix, depth_info = None, **kwargs):
    render_res, depth_t = self.insertor.render_object(
      self.model_bbox, self.normal, self.depth, sg_res,
      self.cam_pose, self.metal, self.rough, self.albedo, **kwargs)
    
    render_res = render_res / (1+render_res)
    render_res = torch.pow(render_res, 1.0/2.2).clip(0,1).cpu().numpy()
    render_res = cv2.cvtColor((render_res*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    msk = np.sum(render_res, axis=-1) != 0
    if depth_info is not None:
      depth_info = cv2.resize(depth_info, (depth_t.shape[1], depth_t.shape[0]))
      
      def on_change(dep_adj):
        dep_adj = ((dep_adj - 500)/500)*6
        global dep_adj_final
        dep_adj_final = dep_adj
        tm = np.logical_and(
          msk,
          depth_t.cpu().numpy() < depth_info + dep_adj #### add bias ######################
        )
        t = np.copy(env_im)
        t[tm] = render_res[tm]
        cv2.imshow('depth adjust', t)
      cv2.imshow('depth adjust', env_im)
      cv2.createTrackbar('slider', 'depth adjust', 0, 1000, on_change)
      cv2.waitKey(0)

      msk = np.logical_and(
        msk,
        depth_t.cpu().numpy() < depth_info + dep_adj_final #### add bias ######################
      )
    
    env_im[msk] = render_res[msk]
    cv2.imwrite(os.path.join(
      self.insertor.gen_path, 'results', '{}_{}.png'.format(self.save_idx, subfix)), env_im)

  def CmpMethodsDecoder(self, buf):
    ######### IRAdobe method
    mb = self.model_bbox
    model_pos_scc = [
      (mb[0][0]+mb[1][0])/2/self.insertor.H, 
      (mb[0][1]+mb[1][1])/2/self.insertor.W
    ]#hw
    sg_info = np.load(os.path.join(IRAdobe_path, '{}_env_envSGCmp1.npy'.format(self.save_idx)))
    depth_info = np.load(os.path.join(IRAdobe_path, '{}_env_depth1.npy'.format(self.save_idx)))
    sgH, sgW = sg_info.shape[-2], sg_info.shape[-1]
    model_pos_scc[0] = int(model_pos_scc[0]*sgH)
    model_pos_scc[1] = int(model_pos_scc[1]*sgW)

    kwargs = {}
    if self.model_radius != None:
      kwargs = {
        'model_radius': self.model_radius,
        'model_pos': self.model_pos,
        'model_bbox': self.model_bbox,
        'model_bbox_last': self.model_bbox_last,
        'gen_shadow': self.shadow_mode
      }
    if self.s_texSize != None:
      kwargs.update({
        's_texSize': self.s_texSize,
        's_VP': self.s_VP,
        's_im': self.s_im,
      })
    if use_std_sf:
      kwargs.update({'model_rot_inv': self.model_rot_inv})
    sgIR = torch.Tensor(sg_info[..., model_pos_scc[0], model_pos_scc[1]]) # 12,7
    sgIR = trans_raw_sg(sgIR)
    #sgIR[:,2] = -sgIR[:,2] ### try inverse Z axis ??
    show_im_cv(SG2Envmap_forDraw(sgIR, 128, 128).cpu().numpy())

    env_im_path = os.path.join(
      self.insertor.gen_path, 'results', '{}_env.png'.format(self.save_idx))
    env_im = cv2.imread(env_im_path)
    self.CompositeCmpResults(sgIR, env_im, 'IR', depth_info, **kwargs)

    ######## EMLight Method
    os.system('cd {}; python3 test_.py {}'.format(EMLight_path, os.path.abspath(env_im_path)))
    sgEM = torch.load(os.path.join(EMLight_path, 'res.tar'))# 
    sgEM = trans_raw_sg(sgEM)
    show_im_cv(SG2Envmap_forDraw(sgEM, 128, 128).cpu().numpy())

    env_im = cv2.imread(env_im_path)
    self.CompositeCmpResults(sgEM, env_im, 'EM', None, **kwargs)


  def SG_use_sshadow(self, buf):
    use_ss = struct.unpack('i', buf)[0]
    global sg_use_self_shadow
    if use_ss == 1:
      sg_use_self_shadow = True
    else:
      sg_use_self_shadow = False

  def SaveResults(self, buf, **kwargs):
    is_save_infos = struct.unpack('i', buf[:4])[0]
    save_prefix = buf[4:].decode()

    results_path = os.path.join(self.insertor.gen_path, 'results')
    rgb, rgb_HDR, obj_depth, obj_render = self.insertor.render_insert_object(
      self.normal, self.depth, 
      self.cam_pose, self.sg if use_sg_base else self.sh, 
      self.metal, self.rough, self.albedo, True, **kwargs
    )
    res = (np.clip(rgb, 0, 1)*255).astype('uint8')
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    cv2.imwrite(
      os.path.join(results_path, '{}_{}.png'.format(self.save_idx, save_prefix)),
      res)
    
    if is_save_infos == 1:
      torch.save({ # save necessary infos
        "rgb_HDR": rgb_HDR,
        "obj_depth": obj_depth,
        "obj_render": obj_render
      }, os.path.join(results_path, '{}_info.tar'.format(self.save_idx)))
      print('Current render result saved with id: {}'.format(self.save_idx))

    return rgb

  # Render a complete results and then can call it
  def RunDecompositionCmpDecoder(self, buf):
    results_path = os.path.join(self.insertor.gen_path, 'results')

    def toIm(im):
      im = im / (1.0+im)
      im = torch.pow(im, 1.0/2.2)
      im = (im.clip(0, 1)*255).cpu().numpy().astype(np.uint8)
      im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
      return im

    # Save SG map
    SGmap = toIm(SG2Envmap_forDraw(self.sg, 256, 512).flip(0).flip(1)) ###
    cv2.imwrite(os.path.join(results_path, '{}_nerf_SG.png'.format(self.save_idx)), SGmap)

    # Save object render results
    obj_render_im = torch.load(
      os.path.join(results_path, 
      '{}_info.tar'.format(self.save_idx))
    )['obj_render'].reshape(self.insertor.H, self.insertor.W, 3)
    black_mask = torch.sum(obj_render_im, dim=-1) == 0

    obj_render_im = toIm(obj_render_im)
    obj_render_im[black_mask.cpu().numpy()] = 255
    cv2.imwrite(os.path.join(results_path, '{}_nerf_obj.png'.format(self.save_idx)), obj_render_im)

    # Save object insertion without shadow and self-shadow
    global sg_use_self_shadow
    sd = self.shadow_mode
    ssd = sg_use_self_shadow
    self.shadow_mode = False
    sg_use_self_shadow = False
    cbuf = struct.pack('i', 0)+'nerf_no_any_shadow'.encode()
    self.Render(cbuf)

    # Save object insertion without self-shadow
    self.shadow_mode = True
    sg_use_self_shadow = False
    cbuf = struct.pack('i', 0)+'nerf_no_self_shadow'.encode()
    self.Render(cbuf)

    sg_use_self_shadow = True

    # Save no global SH result (with all shadow)
    if self.insertor.global_SH is not None:
      gSH = self.insertor.global_SH
      env_opt_iter = self.insertor.env_opt.N_iter

      self.insertor.env_opt.N_iter = 320
      self.insertor.global_SH = None
      # for there is huge light change, sg should be optimized again with more iters
      self.sg = self.insertor.generate_probe(self.model_pos, False, False)
      self.sg = trans_raw_sg(self.sg)
      cbuf = struct.pack('i', 0)+'nerf_no_globalSH'.encode()
      self.Render(cbuf)
      SGmap = toIm(SG2Envmap_forDraw(self.sg, 256, 512).flip(0).flip(1)) ###
      cv2.imwrite(os.path.join(results_path, '{}_nerf_SG_no_globalSH.png'.format(self.save_idx)), SGmap)
      
      self.insertor.global_SH = gSH
      self.insertor.env_opt.N_iter = env_opt_iter

      sh_env = toIm(SH2Envmap_forDraw(gSH, 256, 512).flip(0).flip(1)) ###
      cv2.imwrite(os.path.join(results_path, '{}_globalSH.png'.format(self.save_idx)), sh_env)

    self.shadow_mode = sd
    sg_use_self_shadow = ssd

  def UpdateSaveIndexDecoder(self, buf):
    cmp_path = os.path.join(self.insertor.gen_path, 'results', 'cmp{}'.format(self.save_idx))
    try:
      os.mkdir(cmp_path)
      os.system('cd {}; mv ../{}_* ./'.format(cmp_path, self.save_idx))
    except:
      print('{} is exist, auto organize close')
    self.save_idx = struct.unpack('i', buf)[0]


  def Render(self, buf):
    t_s = time.time()
    #fps = 0 if self.dt == 0 else int(1/self.dt)
    #title = 'render, fps: %d, dt = %.3f'%(fps, self.dt)
    #print(title)
    
    # if pose changed, re-draw the whole canvas
    if self.pose_last is not None:
      if torch.sum(torch.abs(self.cam_pose - self.pose_last)) > 1e-6:
        self.model_bbox_last = None
    self.pose_last = self.cam_pose
    
    if self.normal == None or self.depth == None or (self.sh == None and self.sg == None):
      if self.cam_pose == None:
        print('Error: render info not complete')
      else:
        rgb,_,_,_ = self.insertor.render_pose(self.cam_pose)
        show_im_cv(rgb)
    else:
      kwargs = {}
      if self.model_radius != None:
        kwargs = {
          'model_radius': self.model_radius,
          'model_pos': self.model_pos,
          'model_bbox': self.model_bbox,
          'model_bbox_last': self.model_bbox_last,
          'gen_shadow': self.shadow_mode
        }
      if self.s_texSize != None:
        kwargs.update({
          's_texSize': self.s_texSize,
          's_VP': self.s_VP,
          's_im': self.s_im,
        })
      if use_std_sf:
        kwargs.update({'model_rot_inv': self.model_rot_inv})

      if len(buf) != 0: ## save results
        rgb = self.SaveResults(buf, **kwargs)
        
      else:
        rgb = self.insertor.render_insert_object(
          self.normal, self.depth, 
          self.cam_pose, self.sg if use_sg_base else self.sh, 
          self.metal, self.rough, self.albedo, False, **kwargs
        )

      show_im_cv(rgb)
      #show_im(torch.Tensor(rgb))
      if self.vw != None:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.vw.write((rgb*255).astype('uint8'))
    
    t_e = time.time()
    self.dt = t_e - t_s
    '''
    self.normal = None; self.depth = None
    self.cam_pose = None
    '''
    self.render_num += 1

    try:
      self.server.send(struct.pack('i', 0)) # send 0 to client, render complete
    except:
      pass

  
  def run(self):
    while True:
      buf = self.server.receive()
      if buf == '': break
      action = int.from_bytes(buf[:4], 'little')
      if action == 0: break
      self.act_dict[action](buf[4:])

  def __del__(self):
    if self.vw != None:
      self.vw.release()


if __name__ == '__main__':
  hparams = get_opts()
  insertor = NGP_Insertor(hparams)
  insertor.generate_point_cloud()
  # if use_sg_base:
  #   insertor.generate_envmaps(512)
  #   insertor.load_or_train_envmaps(1001)
  if not hparams.no_global_SH:
    insertor.train_global_SH_light(False)

  NGP_Server(insertor, False).run()




