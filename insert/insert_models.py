import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
from tqdm import tqdm
from render_utils import *
from insert_utils import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class MLP(nn.Module):
  def __init__(self, D=4, W=128, input_ch=63, output_ch=9, skips=[2]):
    super(MLP, self).__init__()
    self.D = D
    self.W = W
    self.skips = skips
    self.input_ch = input_ch
    self.output_ch = output_ch
    
    self.pts_linears = nn.ModuleList(
      [nn.Linear(input_ch, W)] + 
      [nn.Linear(W, W) if i not in self.skips else 
      nn.Linear(W + input_ch, W) for i in range(D-1)
    ])
    
    self.output_linear = nn.Linear(W, output_ch)

  def forward(self, input_pts):
    h = input_pts
    for i, l in enumerate(self.pts_linears):
      h = self.pts_linears[i](h)
      h = F.relu(h)
      if i in self.skips:
        h = torch.cat([input_pts, h], -1)

    outputs = self.output_linear(h)
    return outputs   

class Embedder:
  def __init__(self, **kwargs):
    self.kwargs = kwargs
    self.create_embedding_fn()
      
  def create_embedding_fn(self):
    embed_fns = []
    d = self.kwargs['input_dims']
    out_dim = 0
    if self.kwargs['include_input']:
        embed_fns.append(lambda x : x)
        out_dim += d
        
    max_freq = self.kwargs['max_freq_log2']
    N_freqs = self.kwargs['num_freqs']
    
    if self.kwargs['log_sampling']:
        freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
    else:
        freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        
    for freq in freq_bands:
        for p_fn in self.kwargs['periodic_fns']:
            embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
            out_dim += d
                
    self.embed_fns = embed_fns
    self.out_dim = out_dim
        
  def embed(self, inputs):
    return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
  if i == -1:
    return nn.Identity(), 3
  
  embed_kwargs = {
    'include_input' : True,
    'input_dims' : 3,
    'max_freq_log2' : multires-1,
    'num_freqs' : multires,
    'log_sampling' : True,
    'periodic_fns' : [torch.sin, torch.cos],
  }
  
  embedder_obj = Embedder(**embed_kwargs)
  embed = lambda x, eo=embedder_obj : eo.embed(x)
  return embed, embedder_obj.out_dim

def create_model(
  input_ch, output_ch, model_path, prefix, SH_num,
  train_mat_sh=False, D=2, W=64, skips=[], **kwargs):

  mlp_model = MLP(input_ch=input_ch, output_ch=output_ch, D=D, W=W, skips=skips)

  lrate = kwargs.get('lrate', 5e-3)
  lrate_decay = kwargs.get('lrate_decay', 250)
  optimizer = torch.optim.Adam(
    params=mlp_model.parameters(), 
    lr=lrate, betas=(0.9, 0.999),
  )

  schedule = torch.optim.lr_scheduler.StepLR(optimizer, lrate_decay, 0.1)

  ckpts = [os.path.join(model_path, f) for f in 
    sorted(os.listdir(model_path)) 
    if ('tar' in f and prefix in f)]

  if len(ckpts) == 0:
    print('Can not find pretrained {} model, new model will be created'.format(prefix))
    sh_init = torch.rand(SH_num,3)*2-1
    sh_init[0,:] = torch.rand(3)
    # NOTE !!: 
    # ensure that SH0 init positive to ensure irradiance init positive
    # negtive init irradiance will encourage albedo be negative, with sigmoid, albedo will be zero!
    # so we use leakyrelu to avoid irradiance be too negative, do not use relu, which will make no grad!
    # tiny init irradiance will also make optimization hardly coverage
    if train_mat_sh:
      global_sh = nn.parameter.Parameter(
        sh_init, requires_grad=True
      )
      optimizer.add_param_group({'params': global_sh})
      return mlp_model, optimizer, schedule, 1, global_sh
    return mlp_model, optimizer, schedule, 1
  else:
    ckpt = torch.load(ckpts[-1])
    print('Load ckpt: '+ckpts[-1])
    if train_mat_sh:
      global_sh = ckpt['global_sh']
      optimizer.add_param_group({'params': global_sh})
    mlp_model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    schedule.load_state_dict(ckpt['schedule'])
    epoch = ckpt['epoch_num']
    if train_mat_sh:
      return mlp_model, optimizer, schedule, epoch, global_sh
    return mlp_model, optimizer, schedule, epoch

def train_global_env(pts, normal, gt, model_save_path, SH_num, visual_mat = False, **kwargs):
  embed_fn, input_ch = get_embedder(2) # 3 -> 15 (3+4*3)

  mlp_mat, optimizer, schedule, epoch_s, global_sh = \
    create_model(input_ch, 3, model_save_path, 'mat_sh', SH_num, train_mat_sh=True)

  nerf_model = kwargs.get('nerf_model', None)

  iter_num = kwargs.get('iters', 20000)
  if epoch_s == iter_num - 1:
    if nerf_model is not None:
      nerf_model.global_SH = global_sh.detach()
    return global_sh.detach()
  
  batch_num = kwargs.get('batch', 20480*16)
  ckpt_save = kwargs.get('ckpt_save', 400)
  mat_smooth_range = kwargs.get('mat_smooth_range', 1e-3)
  mat_smooth_weight = kwargs.get('mat_smooth_weight', 0.2)
  hdr_mapping = kwargs.get('hdr_mapping', False)

  downsample_pts_num = kwargs.get('downsample_pts_num', None)
  
  if nerf_model is not None:
    nerf_model.global_SH = global_sh

  for epoch in range(epoch_s, iter_num):
    if epoch % 50 == 1 or epoch == epoch_s:
      print('DO SHUFFLE')
      pts_num = pts.shape[0]
      shuffle_idx = np.random.permutation(pts_num)
      pts_shf = pts[shuffle_idx]
      gt_shf = gt[shuffle_idx]
      norm_shf = normal[shuffle_idx]

    if downsample_pts_num is not None:
      pts_num = downsample_pts_num
    
    for i in range(0, pts_num, batch_num):
      ed = min(i+batch_num, pts_num)
      pts_batch = torch.Tensor(pts_shf[i:ed])
      gt_batch = torch.Tensor(gt_shf[i:ed])
      norm_batch = torch.Tensor(norm_shf[i:ed])

      _out = mlp_mat(embed_fn(pts_batch))
      albedo = torch.sigmoid(_out)

      # NOTE: In training, do not use ReLU in output !!!!!
      # allow negtive value to keep grad
      if nerf_model is not None:
        raw_rgb, rays_d = nerf_model.\
          generate_SH_probes(pts_batch+norm_batch*0.01, return_raw_rgb = True)
        diff_irradiance = F.leaky_relu(irradiance_numerical(raw_rgb, rays_d, norm_batch, allow_neg=True))

      else:
        pts_sh = global_sh.unsqueeze(0).expand(ed-i,SH_num,3)
        diff_irradiance = F.leaky_relu(SH9_irradiance(norm_batch, pts_sh, allow_neg=True))

      '''
      if nerf_model is not None:
        pts_sh = nerf_model.generate_SH_probes(pts_batch)
      else:
        pts_sh = global_sh.expand(ed-i,SH_num,3)

      diff_irradiance = SH9_irradiance(norm_batch, pts_sh)
      '''

      col = albedo / torch.pi * diff_irradiance
      if hdr_mapping:
        col = col / (1+col)
        col = torch.pow(col, 1.0/2.2)

      optimizer.zero_grad()
      loss_c = F.mse_loss(col, gt_batch)
      # albedo smooth loss
      pts_near = (torch.rand_like(pts_batch)*2-1) * mat_smooth_range
      plane_near = torch.sum(pts_near * norm_batch, dim = -1, keepdim=True)*norm_batch
      plane_near = pts_near + pts_batch - plane_near

      out_near = mlp_mat(embed_fn(plane_near))
      albedo_near = torch.sigmoid(out_near)
      loss_mat = mat_smooth_weight * F.mse_loss(albedo, albedo_near)

      loss = loss_c + loss_mat
      if epoch < iter_num * (0.75) and nerf_model is None:
        loss += 2*F.mse_loss(global_sh, global_sh.mean(dim=-1, keepdim=True))

      loss.backward()
      optimizer.step()
      schedule.step()
    
    print('epoch {}/{}, loss = {:.4f}'.format(epoch, iter_num, loss_c.item()))
    #visualize_SH(global_sh.detach(), 256, hdr = True, use_cv=True)
    if epoch % ckpt_save == 0:

      ckpt_for_save = {
        "model": mlp_mat.state_dict(),
        "optimizer": optimizer.state_dict(),
        "schedule": schedule.state_dict(),
        "epoch_num": epoch,
        "global_sh": global_sh
      }
      path = os.path.join(model_save_path, '{}_{:06d}.tar'.format('mat_sh', epoch))
      torch.save(ckpt_for_save, path)
      print('Ckpt Saved')

  if nerf_model is not None:
    nerf_model.global_SH = global_sh.detach()
  
  if visual_mat:
    irrs = []
    albs = []
    rgbs = []
    if nerf_model is not None:
      pass
      #nerf_model.global_SH = None
    batch_num = 4096#20480*16
    pts_num = pts.shape[0]
    shuffle_idx = np.random.permutation(pts_num)
    pts_shf = pts[shuffle_idx]
    gt_shf = gt[shuffle_idx]
    norm_shf = normal[shuffle_idx]

    pts_num = pts.shape[0] // 10
    for i in tqdm(range(0, pts_num, batch_num)):
      ed = min(i+batch_num, pts_num)
      pts_batch = torch.Tensor(pts_shf[i:ed])
      gt_batch = torch.Tensor(gt_shf[i:ed])
      norm_batch = torch.Tensor(norm_shf[i:ed])

      if nerf_model is not None:
        raw_rgb, rays_d = nerf_model.\
          generate_SH_probes(pts_batch+norm_batch*0.01, return_raw_rgb = True)
        #for c, d in zip(raw_rgb, rays_d):
        #  visualize_env(torch.stack([d, c], 0), 32)
        diff_irradiance = irradiance_numerical(raw_rgb, rays_d, norm_batch)

      else:
        #_out = mlp_mat(embed_fn(pts_batch))
        #albedo = torch.sigmoid(_out)
        pts_sh = global_sh.unsqueeze(0).expand(ed-i,SH_num,3)
        diff_irradiance = SH9_irradiance(norm_batch, pts_sh)

      _out = mlp_mat(embed_fn(pts_batch))
      albedo = torch.sigmoid(_out)
      albs.append(albedo.detach().cpu())
      
      wrgb = 1.0 / torch.pi * diff_irradiance
      wrgb = wrgb / (1+wrgb)
      wrgb = torch.pow(wrgb, 1.0/2.2)
      irrs.append(wrgb.detach().cpu())

      cols = albedo / torch.pi * diff_irradiance
      cols = cols / (1+cols)
      cols = torch.pow(cols, 1.0/2.2)
      rgbs.append(cols.detach().cpu())

    rgbs = torch.concat(rgbs, 0).cpu().numpy()
    irrs = torch.concat(irrs, 0).cpu().numpy()
    albs = torch.concat(albs, 0).cpu().numpy()
    write2ply(rgbs, pts_shf[:pts_num], './pts_visual/mat_rgb_visual.ply')
    write2ply(irrs, pts_shf[:pts_num], './pts_visual/mat_irr_visual.ply')
    write2ply(albs, pts_shf[:pts_num], './pts_visual/mat_alb_visual.ply')

  return global_sh.detach()


def train_global_env_prec(pts, normal, gt, rgb_shs, opc_shs,
  model_save_path, SH_num, visual_mat = False, **kwargs):
  embed_fn, input_ch = get_embedder(4) # 3 -> 27 (3+8*3)

  mlp_mat, optimizer, schedule, epoch_s, global_sh = \
    create_model(input_ch, 3, model_save_path, 'mat_sh', SH_num, train_mat_sh=True)

  iter_num = kwargs.get('iters', 20000)
  if epoch_s == iter_num - 1:
    return global_sh.detach()
  
  batch_num = kwargs.get('batch', 20480*16)
  ckpt_save = kwargs.get('ckpt_save', 400)
  mat_smooth_range = kwargs.get('mat_smooth_range', 1e-3)
  mat_smooth_weight = kwargs.get('mat_smooth_weight', 0.2)
  hdr_mapping = kwargs.get('hdr_mapping', False)

  downsample_pts_num = kwargs.get('downsample_pts_num', None)

  SH_tp = SH9_Triple_Product()

  for epoch in range(epoch_s, iter_num):
    if epoch % 50 == 1 or epoch == epoch_s:
      print('DO SHUFFLE')
      pts_num = pts.shape[0]
      shuffle_idx = np.random.permutation(pts_num)
      pts_shf = pts[shuffle_idx]
      gt_shf = gt[shuffle_idx]
      norm_shf = normal[shuffle_idx]
      if rgb_shs is not None:
        rgb_shs_shf = rgb_shs[shuffle_idx]
        opc_shs_shf = opc_shs[shuffle_idx]

    if downsample_pts_num is not None:
      pts_num = downsample_pts_num
    
    for i in range(0, pts_num, batch_num):
      ed = min(i+batch_num, pts_num)
      pts_batch = torch.Tensor(pts_shf[i:ed])
      gt_batch = torch.Tensor(gt_shf[i:ed])
      norm_batch = torch.Tensor(norm_shf[i:ed])
      if rgb_shs is not None:
        rgb_shs_batch = torch.Tensor(rgb_shs_shf[i:ed])
        opc_shs_batch = torch.Tensor(opc_shs_shf[i:ed])

      _out = mlp_mat(embed_fn(pts_batch))
      albedo = torch.sigmoid(_out)

      pts_sh = global_sh.unsqueeze(0).expand(ed-i,SH_num,3)
      if rgb_shs is not None:
        lg_shs = rgb_shs_batch + SH_tp.SH9_product_93(pts_sh, opc_shs_batch)
        diff_irradiance = F.leaky_relu(SH9_irradiance(norm_batch, lg_shs, allow_neg=True))

      else:
        diff_irradiance = F.leaky_relu(SH9_irradiance(norm_batch, pts_sh, allow_neg=True))

      '''
      if nerf_model is not None:
        pts_sh = nerf_model.generate_SH_probes(pts_batch)
      else:
        pts_sh = global_sh.expand(ed-i,SH_num,3)

      diff_irradiance = SH9_irradiance(norm_batch, pts_sh)
      '''

      col = albedo / torch.pi * diff_irradiance
      if hdr_mapping:
        col = col / (1+col)
        col = torch.pow(col, 1.0/2.2)

      optimizer.zero_grad()
      # use tanh, we focus on shadow, not high light
      # to make looks good, not absolutely precise
      loss_c = F.mse_loss(torch.tanh(col), torch.tanh(gt_batch))
      # albedo smooth loss
      pts_near = (torch.rand_like(pts_batch)*2-1) * mat_smooth_range
      plane_near = torch.sum(pts_near * norm_batch, dim = -1, keepdim=True)*norm_batch
      plane_near = pts_near + pts_batch - plane_near

      out_near = mlp_mat(embed_fn(plane_near))
      albedo_near = torch.sigmoid(out_near)
      loss_mat = mat_smooth_weight * F.mse_loss(albedo, albedo_near)

      loss = loss_c + loss_mat
      if epoch < iter_num * (0.75):
        loss += 2*F.mse_loss(global_sh, global_sh.mean(dim=-1, keepdim=True))

      loss.backward()
      optimizer.step()
      schedule.step()
    
    print('epoch {}/{}, loss = {:.4f}'.format(epoch, iter_num, loss_c.item()))
    #visualize_SH(global_sh.detach(), 256, hdr = True, use_cv=True)
    if epoch % ckpt_save == 0:

      ckpt_for_save = {
        "model": mlp_mat.state_dict(),
        "optimizer": optimizer.state_dict(),
        "schedule": schedule.state_dict(),
        "epoch_num": epoch,
        "global_sh": global_sh
      }
      path = os.path.join(model_save_path, '{}_{:06d}.tar'.format('mat_sh', epoch))
      torch.save(ckpt_for_save, path)
      print('Ckpt Saved')
  
  if visual_mat:
    irrs = []
    albs = []
    rgbs = []

    batch_num = 20480*16
    pts_num = pts.shape[0]
    shuffle_idx = np.random.permutation(pts_num)
    pts_shf = pts[shuffle_idx]
    gt_shf = gt[shuffle_idx]
    norm_shf = normal[shuffle_idx]
    if rgb_shs is not None:
      rgb_shs_shf = rgb_shs[shuffle_idx]
      opc_shs_shf = opc_shs[shuffle_idx]

    #pts_num = pts.shape[0] // 10
    for i in tqdm(range(0, pts_num, batch_num)):
      ed = min(i+batch_num, pts_num)
      pts_batch = torch.Tensor(pts_shf[i:ed])
      gt_batch = torch.Tensor(gt_shf[i:ed])
      norm_batch = torch.Tensor(norm_shf[i:ed])
      if rgb_shs is not None:
        rgb_shs_batch = torch.Tensor(rgb_shs_shf[i:ed])
        opc_shs_batch = torch.Tensor(opc_shs_shf[i:ed])

      pts_sh = global_sh.unsqueeze(0).expand(ed-i,SH_num,3)
      if rgb_shs is not None:
        lg_shs = rgb_shs_batch + SH_tp.SH9_product_93(pts_sh, opc_shs_batch)
        diff_irradiance = F.leaky_relu(SH9_irradiance(norm_batch, lg_shs, allow_neg=True))

      else:
        diff_irradiance = F.leaky_relu(SH9_irradiance(norm_batch, pts_sh, allow_neg=True))

      _out = mlp_mat(embed_fn(pts_batch))
      albedo = torch.sigmoid(_out)
      albs.append(albedo.detach().cpu())
      
      wrgb = 1.0 / torch.pi * diff_irradiance
      wrgb = wrgb / (1+wrgb)
      wrgb = torch.pow(wrgb, 1.0/2.2)
      irrs.append(wrgb.detach().cpu())

      cols = albedo / torch.pi * diff_irradiance
      cols = cols / (1+cols)
      cols = torch.pow(cols, 1.0/2.2)
      rgbs.append(cols.detach().cpu())

    rgbs = torch.concat(rgbs, 0).cpu().numpy()
    irrs = torch.concat(irrs, 0).cpu().numpy()
    albs = torch.concat(albs, 0).cpu().numpy()
    write2ply(rgbs, pts_shf[:pts_num], './pts_visual/mat_rgb_visual.ply')
    write2ply(irrs, pts_shf[:pts_num], './pts_visual/mat_irr_visual.ply')
    write2ply(albs, pts_shf[:pts_num], './pts_visual/mat_alb_visual.ply')

  return global_sh.detach()