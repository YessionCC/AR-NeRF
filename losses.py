import torch
from torch import nn
import torch.nn.functional as F
import vren


class DistortionLoss(torch.autograd.Function):
    """
    Distortion loss proposed in Mip-NeRF 360 (https://arxiv.org/pdf/2111.12077.pdf)
    Implementation is based on DVGO-v2 (https://arxiv.org/pdf/2206.05085.pdf)

    Inputs:
        ws: (N) sample point weights
        deltas: (N) considered as intervals
        ts: (N) considered as midpoints
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]

    Outputs:
        loss: (N_rays)
    """
    @staticmethod
    def forward(ctx, ws, deltas, ts, rays_a):
        loss, ws_inclusive_scan, wts_inclusive_scan = \
            vren.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan,
                              ws, deltas, ts, rays_a)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        (ws_inclusive_scan, wts_inclusive_scan,
        ws, deltas, ts, rays_a) = ctx.saved_tensors
        dL_dws = vren.distortion_loss_bw(dL_dloss, ws_inclusive_scan,
                                         wts_inclusive_scan,
                                         ws, deltas, ts, rays_a)
        return dL_dws, None, None, None


class NeRFLoss(nn.Module):
    def __init__(self, epoch, loss_set, grid_scale, lambda_depth, lambda_opacity=1e-3, lambda_distortion=1e-3):
        super().__init__()

        self.num_epoch = epoch
        self.grid_scale = grid_scale
        self.lambda_opacity = lambda_opacity
        self.lambda_depth = lambda_depth
        self.lambda_distortion = lambda_distortion
        # raw loss
        raw_loss = lambda x_est, x_gt: (x_est - x_gt)/(x_est.detach()+1e-3)
        log_loss = lambda x_est, x_gt: torch.log((0.2935+x_est)/(0.2935+x_gt))*0.7607
        tanh_loss = lambda x_est, x_gt: torch.tanh(x_est) - torch.tanh(x_gt)
        if loss_set == 'raw':
            self.rgb_loss = raw_loss
        elif loss_set == 'log':
            self.rgb_loss = log_loss
        elif loss_set == 'tanh':
            self.rgb_loss = tanh_loss
        else:
            print('Unknown loss function!')

    def forward(self, results, target, **kwargs):
        d = {}
        #d['rgb'] = (results['rgb']-target['rgb'])**2
        d['rgb'] = self.rgb_loss(results['rgb'], target['rgb'])**2

        o = results['opacity']+1e-10
        # encourage opacity to be either 0 or 1 to avoid floater
        #cur_step = kwargs['step']
        #if cur_step < self.num_epoch/2*1000:
        d['opacity'] = self.lambda_opacity*(-o*torch.log(o))

        # depth loss, encourage depth large
        d['depth'] = -self.lambda_depth * torch.log((results['depth']/self.grid_scale + 1e-10).clip(max = 1.0))

        if self.lambda_distortion > 0:
            d['distortion'] = self.lambda_distortion * \
                DistortionLoss.apply(results['ws'], results['deltas'],
                                     results['ts'], results['rays_a'])

        return d
