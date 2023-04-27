import torch
from .custom_functions import \
    RayAABBIntersector, RayMarcher, VolumeRenderer
from einops import rearrange
import vren

from insert.insert_utils import *

MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01


@torch.cuda.amp.autocast()
def render(model, rays_o, rays_d, **kwargs):
    """
    Render rays by
    1. Compute the intersection of the rays with the scene bounding box
    2. Follow the process in @render_func (different for train/test)

    Inputs:
        model: NGP
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions

    Outputs:
        result: dictionary containing final rgb and depth
    """
    rays_o = rays_o.contiguous(); rays_d = rays_d.contiguous()
    _, hits_t, _ = \
        RayAABBIntersector.apply(rays_o, rays_d, model.center, model.half_size, 1)
    hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE

    if kwargs.get('test_time', False):
        render_func = __render_rays_test
    else:
        render_func = __render_rays_train

    mesh_depth_map = kwargs.get('mesh_depth_map', None)
    if mesh_depth_map != None: # flattened
        valid_depth = mesh_depth_map >= 1e-6
        hits_t_s = hits_t[valid_depth]
        update_min = torch.min(hits_t_s[:,0,1], mesh_depth_map[valid_depth])
        update_min = torch.max(update_min, hits_t_s[:,0,0])
        hits_t[valid_depth,0,1] = update_min
            

    results = render_func(model, rays_o, rays_d, hits_t, **kwargs)
    for k, v in results.items():
        if kwargs.get('to_cpu', False):
            v = v.cpu()
            if kwargs.get('to_numpy', False):
                v = v.numpy()
        results[k] = v
    return results


@torch.no_grad()
def _backup__render_rays_test(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Render rays by

    while (a ray hasn't converged)
        1. Move each ray to its next occupied @N_samples (initially 1) samples 
           and evaluate the properties (sigmas, rgbs) there
        2. Composite the result to output; if a ray has transmittance lower
           than a threshold, mark this ray as converged and stop marching it.
           When more rays are dead, we can increase the number of samples
           of each marching (the variable @N_samples)
    """
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    results = {}

    # output tensors to be filled in
    N_rays = len(rays_o)
    device = rays_o.device
    opacity = torch.zeros(N_rays, device=device)
    depth = torch.zeros(N_rays, device=device)
    rgb = torch.zeros(N_rays, 3, device=device)

    samples = total_samples = 0
    alive_indices = torch.arange(N_rays, device=device)
    # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
    # otherwise, 4 is more efficient empirically
    min_samples = 1 if exp_step_factor==0 else 4

    while samples < kwargs.get('max_samples', MAX_SAMPLES):
        N_alive = len(alive_indices)
        if N_alive==0: break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays//N_alive, 64), min_samples)
        samples += N_samples

        xyzs, dirs, deltas, ts, N_eff_samples = \
            vren.raymarching_test(rays_o, rays_d, hits_t[:, 0], alive_indices,
                                  model.density_bitfield, model.cascades,
                                  model.scale, exp_step_factor,
                                  model.grid_size, MAX_SAMPLES, N_samples)
        total_samples += N_eff_samples.sum()
        xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
        dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        valid_mask = ~torch.all(dirs==0, dim=1)
        if valid_mask.sum()==0: break

        sigmas = torch.zeros(len(xyzs), device=device)
        rgbs = torch.zeros(len(xyzs), 3, device=device)

        xyzs = xyzs[valid_mask]; dirs = dirs[valid_mask]
        pts_num = xyzs.shape[0]
        val_batch_size = kwargs.get('val_batch_size', pts_num)
        sigma_bat_res = []; rgb_bat_res = []
        for i in range(0, pts_num, val_batch_size):
            sigma_bat, rgb_bat = model(xyzs[i:i+val_batch_size], dirs[i:i+val_batch_size], **kwargs)
            sigma_bat_res.append(sigma_bat)
            rgb_bat_res.append(rgb_bat)
        sigmas[valid_mask] = torch.concat(sigma_bat_res, 0)
        rgbs[valid_mask] = torch.concat(rgb_bat_res, 0).float()
        #sigmas[valid_mask], _rgbs = model(xyzs[valid_mask], dirs[valid_mask], **kwargs)
        #rgbs[valid_mask] = _rgbs.float()

        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)

        vren.composite_test_fw(
            sigmas, rgbs, deltas, ts,
            hits_t[:, 0], alive_indices, kwargs.get('T_threshold', 1e-4),
            N_eff_samples, opacity, depth, rgb)
        alive_indices = alive_indices[alive_indices>=0] # remove converged rays

    results['opacity'] = opacity
    results['depth'] = depth
    results['rgb'] = rgb
    results['total_samples'] = total_samples # total samples for all rays

    rgb_bg = torch.zeros(3, device=device)
    with torch.enable_grad():
        '''
        if exp_step_factor==0: # synthetic
            rgb_bg = torch.ones(3, device=device)
        else: # real
            rgb_bg = torch.zeros(3, device=device)
        '''
        # SH bkg for global light optimize
        SH_bkg = kwargs.get('SH_bkg', None) # enable grad to optimize SH bkg
        if SH_bkg != None:
            # if SH in training, not clamp to positive to keep grad
            # else clamp to avoid negtive value
            rgb_bg = get_SH_val(SH_bkg, rays_d, 
                clamp_postive=not SH_bkg.requires_grad)
        
        # Image bkg for virtual object merge to nerf
        IM_bkg = kwargs.get('IM_bkg', None)
        if IM_bkg != None:
            rgb_bg = IM_bkg

        if kwargs.get('blend_bkg', True):
            results['rgb'] += rgb_bg*rearrange(1-opacity, 'n -> n 1')

    #del rgb_bg
    return results

@torch.no_grad()
def __render_rays_test(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Render rays by

    while (a ray hasn't converged)
        1. Move each ray to its next occupied @N_samples (initially 1) samples 
           and evaluate the properties (sigmas, rgbs) there
        2. Composite the result to output; if a ray has transmittance lower
           than a threshold, mark this ray as converged and stop marching it.
           When more rays are dead, we can increase the number of samples
           of each marching (the variable @N_samples)
    """
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    results = {}

    # output tensors to be filled in
    N_rays = len(rays_o)
    device = rays_o.device
    opacity = torch.zeros(N_rays, device=device)
    depth = torch.zeros(N_rays, device=device)
    rgb = torch.zeros(N_rays, 3, device=device)

    samples = total_samples = 0
    alive_indices = torch.arange(N_rays, device=device)
    # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
    # otherwise, 4 is more efficient empirically
    min_samples = 1 if exp_step_factor==0 else 4

    while samples < kwargs.get('max_samples', MAX_SAMPLES):
        N_alive = len(alive_indices)
        if N_alive==0: break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays//N_alive, 64), min_samples)
        samples += N_samples

        xyzs, dirs, deltas, ts, N_eff_samples = \
            vren.raymarching_test(rays_o, rays_d, hits_t[:, 0], alive_indices,
                                  model.density_bitfield, model.cascades,
                                  model.scale, exp_step_factor,
                                  model.grid_size, MAX_SAMPLES, N_samples)
        total_samples += N_eff_samples.sum()
        xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
        dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        valid_mask = ~torch.all(dirs==0, dim=1)
        if valid_mask.sum()==0: break

        sigmas = torch.zeros(len(xyzs), device=device)
        rgbs = torch.zeros(len(xyzs), 3, device=device)

        xyzs = xyzs[valid_mask]; dirs = dirs[valid_mask]
        pts_num = xyzs.shape[0]
        val_batch_size = kwargs.get('val_batch_size', pts_num)
        sigma_bat_res = []; rgb_bat_res = []
        for i in range(0, pts_num, val_batch_size):
            sigma_bat, rgb_bat = model(xyzs[i:i+val_batch_size], dirs[i:i+val_batch_size], **kwargs)
            sigma_bat_res.append(sigma_bat)
            rgb_bat_res.append(rgb_bat)
        sigmas[valid_mask] = torch.concat(sigma_bat_res, 0)
        rgbs[valid_mask] = torch.concat(rgb_bat_res, 0).float()
        #sigmas[valid_mask], _rgbs = model(xyzs[valid_mask], dirs[valid_mask], **kwargs)
        #rgbs[valid_mask] = _rgbs.float()

        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)

        vren.composite_test_fw(
            sigmas, rgbs, deltas, ts,
            hits_t[:, 0], alive_indices, kwargs.get('T_threshold', 1e-4),
            N_eff_samples, opacity, depth, rgb)
        alive_indices = alive_indices[alive_indices>=0] # remove converged rays

    results['opacity'] = opacity
    results['depth'] = depth
    results['rgb'] = rgb
    results['total_samples'] = total_samples # total samples for all rays

    rgb_bg = torch.zeros(3, device=device)
    SH_bkg = kwargs.get('SH_bkg', None) 
    if SH_bkg != None:
        rgb_bg = get_SH_val(SH_bkg, rays_d, clamp_postive=True)
    
    IM_bkg = kwargs.get('IM_bkg', None)
    if IM_bkg != None:
        rgb_bg = IM_bkg

    if kwargs.get('blend_bkg', True):
        results['rgb'] += rgb_bg*rearrange(1-opacity, 'n -> n 1')

    #del rgb_bg
    return results

def __render_rays_train(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Render rays by
    1. March the rays along their directions, querying @density_bitfield
       to skip empty space, and get the effective sample points (where
       there is object)
    2. Infer the NN at these positions and view directions to get properties
       (currently sigmas and rgbs)
    3. Use volume rendering to combine the result (front to back compositing
       and early stop the ray if its transmittance is below a threshold)
    """
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    results = {}

    (rays_a, xyzs, dirs,
    results['deltas'], results['ts'], results['rm_samples']) = \
        RayMarcher.apply(
            rays_o, rays_d, hits_t[:, 0], model.density_bitfield,
            model.cascades, model.scale,
            exp_step_factor, model.grid_size, MAX_SAMPLES)

    for k, v in kwargs.items(): # supply additional inputs, repeated per ray
        if isinstance(v, torch.Tensor):
            kwargs[k] = torch.repeat_interleave(v[rays_a[:, 0]], rays_a[:, 2], 0)
    sigmas, rgbs = model(xyzs, dirs, **kwargs)

    (results['vr_samples'], results['opacity'],
    results['depth'], results['rgb'], results['ws']) = \
        VolumeRenderer.apply(sigmas, rgbs.contiguous(), results['deltas'], results['ts'],
                             rays_a, kwargs.get('T_threshold', 1e-4))
    results['rays_a'] = rays_a

    if kwargs.get('random_bg', False):
        rgb_bg = torch.rand(3, device=rays_o.device)
    else:
        if exp_step_factor==0: # synthetic 
            rgb_bg = torch.ones(3, device=rays_o.device)
        else: # real
            rgb_bg = torch.zeros(3, device=rays_o.device)

    results['rgb'] = results['rgb'] + \
                     rgb_bg*rearrange(1-results['opacity'], 'n -> n 1')
    del rgb_bg
    return results

@torch.enable_grad()
def render_surface_normal(model, pts, **kwargs):
    H,W,_ = pts.shape
    pts_grad = torch.tensor(pts.reshape(-1,3), requires_grad=True)
    sigmas = model.density(pts_grad)
    #loss = torch.mean(sigmas)
    #loss.backward()
    normals = torch.autograd.grad(sigmas, pts_grad, torch.ones_like(sigmas))[0]
    normals = normals.reshape(H,W,3).nan_to_num(0.0, 1.0, -1.0).detach()
    normals = -normalize_eps(normals)
    #show_im(normals.detach()*0.5+0.5)
    #t = torch.nonzero(torch.isnan(normals).any(-1).flatten().int())[:,0]
    #kk = pts.reshape(-1,3)[t]
    return normals

@torch.no_grad()
def render_surface_rgb(model, pts, rays_d, **kwargs):
    H,W,_ = pts.shape
    sigmas, rgbs = model(pts.reshape(-1,3), rays_d.reshape(-1,3), **kwargs)
    rgbs = rgbs.reshape(H,W,3)
    return rgbs